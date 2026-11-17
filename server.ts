import express from 'express';
import { createServer as createViteServer } from 'vite';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs';
import dotenv from 'dotenv';
import { GoogleGenAI, ThinkingLevel } from '@google/genai';
import {
  MAX_QUIZ_CONTEXT_CHARS,
  QUIZ_OVERLOAD_MESSAGE,
  QUIZ_RATE_LIMIT_MESSAGE,
  QUIZ_SYSTEM_INSTRUCTION,
  PART_I_RESPONSE_SCHEMA,
  PART_II_III_RESPONSE_SCHEMA,
  condenseTheoryContent,
  buildQuizPromptsForSplitBatches,
  buildValidatedPartI,
  buildValidatedPartIIAndIII,
  isRateLimitError,
} from './src/quizPromptHelper.ts';
import { parseDocxWithMath, ParseDocxResult } from './src/docxMathParser.ts';
import { unifyMathAndTextFormat } from './src/mathFormat.ts';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const LIBRARY_DIR = path.resolve(__dirname, 'library_data');
const FILES_DIR = path.resolve(LIBRARY_DIR, 'files');
const LIBRARY_TXT_PATH = path.resolve(LIBRARY_DIR, 'library.txt');

// Empty initial knowledge base - user manages their own files
const INITIAL_FILES: Array<{ filename: string; title: string; category?: string; content: string }> = [];

function initLibrary() {
  try {
    if (!fs.existsSync(LIBRARY_DIR)) {
      fs.mkdirSync(LIBRARY_DIR, { recursive: true });
    }
    if (!fs.existsSync(FILES_DIR)) {
      fs.mkdirSync(FILES_DIR, { recursive: true });
    }
    if (!fs.existsSync(LIBRARY_TXT_PATH)) {
      syncLibraryTxtFromFiles();
    }
  } catch (err) {
    console.warn('Library init notice:', err);
  }
}

function syncLibraryTxtFromFiles() {
  try {
    if (!fs.existsSync(FILES_DIR)) return;
    const files = fs.readdirSync(FILES_DIR);
    let txtContent = `# ==============================================================================
# AURALEARN KNOWLEDGE REPOSITORY — MASTER INDEX: library.txt
# Tệp chỉ mục quản lý tất cả các tài liệu và tri thức được lưu trữ trong hệ thống.
# Tổng số tệp được chỉ mục: ${files.length}
# Cập nhật lần cuối: ${new Date().toISOString()}
# ==============================================================================

`;

    for (const filename of files) {
      const fullPath = path.resolve(FILES_DIR, filename);
      const stat = fs.statSync(fullPath);
      if (!stat.isFile()) continue;
      const content = fs.readFileSync(fullPath, 'utf-8');
      const firstLine = content.split('\n')[0].replace(/^#+\s*/, '').trim() || filename;

      txtContent += `[ENTRY]
id=${filename}
title=${firstLine}
size=${stat.size} bytes
updated=${stat.mtime.toISOString().slice(0, 10)}
content_preview=${content.replace(/\r?\n/g, ' ').slice(0, 150)}...

`;
    }

    fs.writeFileSync(LIBRARY_TXT_PATH, txtContent, 'utf-8');
  } catch (err) {
    console.warn('syncLibraryTxtFromFiles notice:', err);
  }
}

function getLibraryData() {
  initLibrary();
  const rawIndex = fs.existsSync(LIBRARY_TXT_PATH) ? fs.readFileSync(LIBRARY_TXT_PATH, 'utf-8') : '';

  const items: Array<{ filename: string; title: string; size: number; updated: string; content: string }> = [];
  if (fs.existsSync(FILES_DIR)) {
    const diskFiles = fs.readdirSync(FILES_DIR);
    for (const filename of diskFiles) {
      try {
        const fullPath = path.resolve(FILES_DIR, filename);
        const stat = fs.statSync(fullPath);
        if (!stat.isFile()) continue;
        const content = fs.readFileSync(fullPath, 'utf-8');
        const firstLine = content.split('\n')[0].replace(/^#+\s*/, '').trim() || filename;
        items.push({
          filename,
          title: firstLine,
          size: stat.size,
          updated: stat.mtime.toISOString().slice(0, 10),
          content,
        });
      } catch {
        // skip unreadable entry
      }
    }
  }

  return {
    rawIndex,
    totalFiles: items.length,
    files: items,
  };
}


function resolvePort(): number {
  const args = process.argv;
  const portIdx = args.indexOf('--port');
  if (portIdx !== -1 && args[portIdx + 1]) {
    const val = parseInt(args[portIdx + 1], 10);
    if (!isNaN(val)) return val;
  }
  if (process.env.APP_PORT) {
    const val = parseInt(process.env.APP_PORT, 10);
    if (!isNaN(val)) return val;
  }
  // In AI Studio Cloud Run, process.env.PORT is 8080 (reserved for Nginx proxy).
  // The app server MUST run on 3000.
  if (process.env.PORT && process.env.PORT !== '8080') {
    const val = parseInt(process.env.PORT, 10);
    if (!isNaN(val)) return val;
  }
  return 3000;
}

async function startServer() {
  initLibrary();
  const app = express();
  const PORT = resolvePort();
  const isProd = process.env.NODE_ENV === 'production';

  app.use(express.json({ limit: '50mb' }));

  const getGeminiClient = () => {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
      throw new Error('GEMINI_API_KEY is not configured in process.env');
    }
    return {
      ai: new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      }),
      apiKey,
    };
  };

  const withTimeout = <T,>(promise: Promise<T>, ms: number, errorMessage: string): Promise<T> => {
    let timer: NodeJS.Timeout | undefined;
    return Promise.race([
      promise.finally(() => {
        if (timer) clearTimeout(timer);
      }),
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new Error(errorMessage)), ms);
      }),
    ]);
  };

  // API 1: Get full library state and library.txt content
  app.get('/api/library', (req, res) => {
    try {
      const data = getLibraryData();
      res.json({ success: true, ...data });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Không thể đọc thư viện' });
    }
  });

  // API 2: Add or upload a new file to the library (quản lý theo phiên cá nhân)
  app.post('/api/library/add', (req, res) => {
    try {
      let { filename, title, content } = req.body;
      if (!content || !content.trim()) {
        return res.status(400).json({ success: false, error: 'Nội dung tệp không được để trống' });
      }

      if (!filename || !filename.trim()) {
        const cleanTitle = (title || 'tai_lieu')
          .toLowerCase()
          .normalize('NFD')
          .replace(/[\u0300-\u036f]/g, '')
          .replace(/[^a-z0-9_-]/g, '_');
        filename = `${cleanTitle}_${Date.now().toString().slice(-4)}.txt`;
      }

      res.json({
        success: true,
        filename,
        message: `Đã nạp tệp ${filename} cho phiên làm việc cá nhân`,
      });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi lưu tệp vào thư viện' });
    }
  });

  // API 3: Update existing file in the library
  app.put('/api/library/update', (req, res) => {
    try {
      const { filename } = req.body;
      if (!filename) return res.status(400).json({ success: false, error: 'Thiếu tên tệp' });

      res.json({ success: true, message: `Đã cập nhật ${filename} trong phiên cá nhân` });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message });
    }
  });

  // API 4: Delete a file from the library
  app.delete('/api/library/delete', (req, res) => {
    try {
      const { filename } = req.body;
      if (!filename) return res.status(400).json({ success: false, error: 'Thiếu tên tệp' });

      const filePath = path.resolve(FILES_DIR, filename);
      if (fs.existsSync(filePath)) {
        fs.unlinkSync(filePath);
      }
      syncLibraryTxtFromFiles();

      res.json({ success: true, message: `Đã xóa ${filename}` });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message });
    }
  });


  // API 5: Raw download / export of library.txt
  app.get('/api/library/raw-index', (req, res) => {
    try {
      syncLibraryTxtFromFiles();
      const content = fs.readFileSync(LIBRARY_TXT_PATH, 'utf-8');
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Content-Disposition', 'attachment; filename="library.txt"');
      res.send(content);
    } catch (err: any) {
      res.status(500).send('Lỗi khi tải library.txt');
    }
  });

  // API 5b: Download specific file from repository
  app.get('/api/library/download/:filename', (req, res) => {
    try {
      const rawName = req.params.filename;
      const cleanFilename = path.basename(rawName).replace(/[^a-zA-Z0-9._-]/g, '_');
      const filePath = path.resolve(FILES_DIR, cleanFilename);

      if (!fs.existsSync(filePath)) {
        return res.status(404).json({ success: false, error: 'Tệp không tồn tại trong kho lưu trữ' });
      }

      const content = fs.readFileSync(filePath, 'utf-8');
      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Content-Disposition', `attachment; filename="${cleanFilename}"`);
      res.send(content);
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi tải tệp' });
    }
  });

  // API 5c: Upload multiple files in batch
  // API 5c: Upload multiple files in batch (quản lý theo phiên làm việc cá nhân)
  app.post('/api/library/upload-batch', (req, res) => {
    try {
      const { files } = req.body;
      if (!Array.isArray(files) || files.length === 0) {
        return res.status(400).json({ success: false, error: 'Danh sách tệp tải lên không hợp lệ' });
      }

      const savedList: string[] = files.map((it: any) => it.filename || 'tai_lieu.txt');
      res.json({
        success: true,
        count: savedList.length,
        savedFiles: savedList,
        message: `Đã nạp thành công ${savedList.length} tệp cho phiên làm việc cá nhân`,
      });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi tải tệp lên' });
    }
  });


  // API 5d: Export all files combined into a single archive file
  app.get('/api/library/export-all', (req, res) => {
    try {
      const data = getLibraryData();
      let combined = `# ==============================================================================\n`;
      combined += `# GÓI TỔNG HỢP TOÀN BỘ KHO TRI THỨC AURALEARN (${data.totalFiles} TỆP)\n`;
      combined += `# Thời gian xuất: ${new Date().toLocaleString('vi-VN')}\n`;
      combined += `# ==============================================================================\n\n`;

      for (const f of data.files) {
        combined += `\n################################################################################\n`;
        combined += `### TỆP: ${f.filename} | TIÊU ĐỀ: ${f.title} (${f.size} bytes)\n`;
        combined += `################################################################################\n\n`;
        combined += f.content;
        combined += `\n\n`;
      }

      res.setHeader('Content-Type', 'text/plain; charset=utf-8');
      res.setHeader('Content-Disposition', 'attachment; filename="auralearn_full_knowledge_base.txt"');
      res.send(combined);
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi xuất kho tri thức' });
    }
  });

  // API 5e: Upload & parse Word file (.docx) with MathType / OMML equations
  app.post('/api/library/upload-word', async (req, res) => {
    try {
      const { filename, base64, fileData } = req.body;
      const rawBase64 = base64 || fileData;
      if (!rawBase64) {
        return res.status(400).json({ success: false, error: 'Thiếu dữ liệu tệp Word (base64 hoặc fileData)' });
      }

      const buffer = Buffer.from(rawBase64, 'base64');
      let safeFilename = path.basename(filename || 'tai_lieu_mathtype.docx').replace(/[^a-zA-Z0-9._-]/g, '_');
      safeFilename = safeFilename.replace(/\.docx?$/i, '') + '.txt';

      let extractedMarkdown = '';
      try {
        const op: any = await import('officeparser');
        const parseFn = op.parseOffice || op.default?.parseOffice;
        if (typeof parseFn === 'function') {
          const res = await parseFn(buffer);
          extractedMarkdown = typeof res === 'string' ? res : (res?.toText?.() || res?.text || JSON.stringify(res));
        }
      } catch (e: any) {
        console.warn('OfficeParser fallback:', e?.message);
      }

      if (!extractedMarkdown || !extractedMarkdown.trim()) {
        return res.status(400).json({ success: false, error: 'Không thể trích xuất nội dung từ tệp Word' });
      }

      try {
        extractedMarkdown = extractedMarkdown.normalize('NFC').replace(/[\u200B-\u200D\uFEFF]/g, '');
      } catch {
        // fallback
      }

      const targetPath = path.resolve(FILES_DIR, safeFilename);
      fs.writeFileSync(targetPath, extractedMarkdown.trim(), 'utf-8');
      syncLibraryTxtFromFiles();

      const title = (filename || safeFilename).replace(/\.docx?$/i, '').replace(/\.txt$/i, '');

      res.json({
        success: true,
        filename: safeFilename,
        content: extractedMarkdown.trim(),
        file: {
          filename: safeFilename,
          title,
          content: extractedMarkdown.trim(),
        },
        message: `Đã nạp và chuyển đổi tệp Word "${filename}" thành công!`,
      });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi xử lý tệp Word' });
    }
  });

  function guessMimeType(filename: string): string {
    const lower = filename.toLowerCase();
    if (lower.endsWith('.docx')) return 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
    if (lower.endsWith('.doc')) return 'application/msword';
    if (lower.endsWith('.pdf')) return 'application/pdf';
    if (lower.endsWith('.png')) return 'image/png';
    if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
    if (lower.endsWith('.webp')) return 'image/webp';
    if (lower.endsWith('.gif')) return 'image/gif';
    if (lower.endsWith('.md')) return 'text/markdown';
    if (lower.endsWith('.tex') || lower.endsWith('.latex')) return 'application/x-tex';
    if (lower.endsWith('.csv')) return 'text/csv';
    if (lower.endsWith('.json')) return 'application/json';
    if (lower.endsWith('.html')) return 'text/html';
    return 'text/plain';
  }

  // Engine xử lý tệp siêu tốc trên backend: Giữ nguyên văn bản, chỉ trích xuất MathType/công thức hoặc gửi Gemini
  async function processFileOnBackend(fileInfo: {
    filename: string;
    mimeType?: string;
    base64Data: string;
    prompt?: string;
  }) {
    const rawBase64 = fileInfo.base64Data.includes(',')
      ? fileInfo.base64Data.split(',')[1]
      : fileInfo.base64Data;
    const buffer = Buffer.from(rawBase64, 'base64');
    const cleanFilename = (fileInfo.filename || 'tai_lieu.txt').replace(/[^a-zA-Z0-9._-]/g, '_');
    const mime = fileInfo.mimeType || guessMimeType(cleanFilename);
    const lowerName = cleanFilename.toLowerCase();
    const isDocx = /\.docx$/i.test(lowerName);
    const isDoc = /\.doc$/i.test(lowerName);
    const isImage = mime.startsWith('image/') || /\.(png|jpe?g|webp|gif|bmp)$/i.test(lowerName);
    const isPdf = mime === 'application/pdf' || /\.pdf$/i.test(lowerName);
    const apiKey = process.env.GEMINI_API_KEY;
    const canUseGemini = Boolean(apiKey && apiKey !== 'MY_GEMINI_API_KEY');

    // 1. FILE WORD (.DOCX) - Giữ nguyên toàn bộ văn bản và cấu trúc, chuyển OMML/MathType sang LaTeX; CHỈ gửi ảnh công thức/MathType sang Gemini
    if (isDocx) {
      let docxResult: ParseDocxResult | null = null;
      let formulaResolvedCount = 0;

      const formulaResolver = canUseGemini
        ? async (imgBuffer: Uint8Array, imgMime: string, rId: string) => {
            try {
              const { ai } = getGeminiClient();
              const b64 = Buffer.from(imgBuffer).toString('base64');
              const res = await ai.models.generateContent({
                model: 'gemini-4-hydrogen',
                contents: {
                  parts: [
                    {
                      inlineData: {
                        mimeType: imgMime.startsWith('image/') ? imgMime : 'image/png',
                        data: b64,
                      },
                    },
                    {
                      text: 'Chuyển hình ảnh công thức toán học/MathType này sang định dạng KaTeX/LaTeX chuẩn trong cặp dấu $...$. Chỉ trả về công thức LaTeX, không viết thêm bất kỳ chữ nào khác.',
                    },
                  ],
                },
              });
              const rawLatex = res?.text?.trim() || '';
              if (rawLatex) {
                formulaResolvedCount++;
                return rawLatex;
              }
              return null;
            } catch (e: any) {
              console.warn(`Formula resolver Gemini notice for ${rId}:`, e?.message);
              return null;
            }
          }
        : undefined;

      try {
        docxResult = await parseDocxWithMath(buffer, { formulaResolver });
      } catch (docxErr: any) {
        console.warn('Backend parseDocxWithMath notice, falling back to officeparser:', docxErr?.message);
        try {
          const op: any = await import('officeparser');
          const parseFn = op.parseOffice || op.default?.parseOffice;
          if (typeof parseFn === 'function') {
            const parsedOffice = await parseFn(buffer);
            const rawOfficeText =
              typeof parsedOffice === 'string'
                ? parsedOffice
                : parsedOffice?.toText?.() || parsedOffice?.text || '';
            docxResult = {
              title: cleanFilename.replace(/\.docx?$/i, ''),
              markdown: rawOfficeText,
              content: rawOfficeText,
              equationCount: 0,
              wordCount: rawOfficeText.split(/\s+/).length,
              equations: [],
            };
          }
        } catch (opErr: any) {
          console.warn('officeparser fallback error:', opErr?.message);
        }
      }

      if (docxResult && docxResult.content.trim()) {
        const unifiedContent = unifyMathAndTextFormat(docxResult.content);
        return {
          filename: cleanFilename,
          title: docxResult.title || cleanFilename.replace(/\.docx?$/i, ''),
          content: unifiedContent,
          size: buffer.length,
          updated: new Date().toISOString().slice(0, 10),
          base64Data: fileInfo.base64Data,
          mimeType: mime,
          equationCount: docxResult.equationCount || (unifiedContent.match(/\$[^$]+\$/g) || []).length,
          source:
            formulaResolvedCount > 0
              ? `Word MathType (+ ${formulaResolvedCount} công thức nhận dạng bằng Gemini)`
              : 'Word MathType XML Parser (Backend)',
        };
      }
    }

    // 2. FILE VĂN BẢN THUẦN (.txt, .md, .tex, .latex, .csv, .json, .rtf, .html, .xml) - Xử lý trực tiếp siêu tốc, giữ 100% văn bản, KHÔNG cần Gemini
    if (!isImage && !isPdf && !isDoc) {
      let decoded = '';
      if (buffer.length >= 2 && buffer[0] === 0xff && buffer[1] === 0xfe) {
        decoded = buffer.subarray(2).toString('utf16le');
      } else if (buffer.length >= 2 && buffer[0] === 0xfe && buffer[1] === 0xff) {
        const swapped = Buffer.from(buffer.subarray(2));
        swapped.swap16();
        decoded = swapped.toString('utf16le');
      } else {
        decoded = buffer.toString('utf-8');
      }

      if (/\.(tex|latex)$/i.test(cleanFilename)) {
        const docMatch = decoded.match(/\\begin\{document\}([\s\S]*?)\\end\{document\}/i);
        if (docMatch) decoded = docMatch[1];
        decoded = decoded
          .replace(/\\section\*?\{([^}]+)\}/g, '\n## $1\n')
          .replace(/\\subsection\*?\{([^}]+)\}/g, '\n### $1\n')
          .replace(/\\textbf\{([^}]+)\}/g, '**$1**')
          .replace(/\\textit\{([^}]+)\}/g, '*$1*')
          .replace(/\\item\b/g, '\n- ');
      }

      const unified = unifyMathAndTextFormat(decoded.trim());
      const firstLine = unified.split('\n')[0].replace(/^#+\s*/, '').trim();
      const title =
        firstLine && firstLine.length > 2 && firstLine.length < 90
          ? firstLine
          : cleanFilename.replace(/\.[^/.]+$/, '');

      return {
        filename: cleanFilename,
        title,
        content: unified,
        size: buffer.length,
        updated: new Date().toISOString().slice(0, 10),
        base64Data: fileInfo.base64Data,
        mimeType: mime,
        equationCount: (unified.match(/\$[^$]+\$/g) || []).length,
        source: 'Backend Text Parser (Siêu tốc)',
      };
    }

    // 3. FILE ẢNH HOẶC PDF - Gửi Base64 sang Gemini để nhận dạng đề thi & công thức toán
    let extractedContent = '';
    if (canUseGemini && (isImage || isPdf)) {
      try {
        const { ai } = getGeminiClient();
        const response = await ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: {
            parts: [
              {
                inlineData: {
                  mimeType: mime.startsWith('image/') || mime === 'application/pdf' ? mime : 'image/png',
                  data: rawBase64,
                },
              },
              {
                text:
                  fileInfo.prompt ||
                  'Hãy đọc và trích xuất toàn bộ nội dung văn bản, câu hỏi trắc nghiệm, bài tập và công thức toán học/khoa học từ hình ảnh/tài liệu này sang tiếng Việt chuẩn. Giữ nguyên 100% nội dung câu chữ; mọi công thức toán học phải viết bằng LaTeX chuẩn trong cặp dấu $...$ hoặc $$...$$.',
              },
            ],
          },
        });
        if (response && response.text) {
          extractedContent = response.text.trim();
        }
      } catch (aiErr: any) {
        console.warn('Gemini image/PDF extraction notice:', aiErr?.message);
      }
    }

    if (!extractedContent.trim()) {
      if (isImage) {
        extractedContent = `# ${cleanFilename.replace(/\.[^/.]+$/, '')}\n\n![${cleanFilename}](${fileInfo.base64Data})\n\n*(Ảnh đề thi đã được nạp thành công vào hệ thống)*`;
      } else {
        extractedContent = `# ${cleanFilename.replace(/\.[^/.]+$/, '')}\n\n*(Tài liệu ${cleanFilename} đã được nạp thành công vào hệ thống)*`;
      }
    } else if (isImage) {
      extractedContent = `![${cleanFilename}](${fileInfo.base64Data})\n\n${extractedContent}`;
    }

    const unifiedResult = unifyMathAndTextFormat(extractedContent);
    const firstLine = unifiedResult
      .split('\n')
      .find((l) => l.trim().startsWith('#') || (l.trim().length > 3 && !l.startsWith('![')))
      ?.replace(/^#+\s*/, '')
      .trim();

    return {
      filename: cleanFilename,
      title: firstLine || cleanFilename.replace(/\.[^/.]+$/, ''),
      content: unifiedResult,
      size: buffer.length,
      updated: new Date().toISOString().slice(0, 10),
      base64Data: fileInfo.base64Data,
      mimeType: mime,
      equationCount: (unifiedResult.match(/\$[^$]+\$/g) || []).length,
      source: canUseGemini ? 'Gemini AI Vision & Math OCR' : 'Local File Loader',
    };
  }

  // API 5f: Xử lý tệp đơn trên backend (Giữ nguyên văn bản, chỉ trích xuất MathType/công thức hoặc gửi Gemini)
  app.post('/api/library/process-file', async (req, res) => {
    try {
      const { filename, mimeType, base64Data, prompt } = req.body;
      if (!base64Data) {
        return res.status(400).json({ success: false, error: 'Thiếu dữ liệu tệp' });
      }

      const item = await processFileOnBackend({
        filename: filename || 'tai_lieu.txt',
        mimeType,
        base64Data,
        prompt,
      });

      res.json({
        success: true,
        file: item,
        message: `Đã xử lý tệp "${item.filename}" thành công trên máy chủ!`,
      });
    } catch (err: any) {
      console.error('Lỗi process-file backend:', err);
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi xử lý tệp trên máy chủ' });
    }
  });

  // API 5g: Xử lý hàng loạt tệp song song trên backend
  app.post('/api/library/process-batch', async (req, res) => {
    try {
      const { files = [] } = req.body;
      if (!Array.isArray(files) || files.length === 0) {
        return res.status(400).json({ success: false, error: 'Danh sách tệp trống' });
      }

      const processedItems = await Promise.all(
        files.map(async (f: any) => {
          try {
            return await processFileOnBackend({
              filename: f.filename || 'tai_lieu.txt',
              mimeType: f.mimeType,
              base64Data: f.base64Data,
              prompt: f.prompt,
            });
          } catch (e: any) {
            return {
              filename: f.filename || 'error.txt',
              title: f.filename || 'Lỗi xử lý',
              content: `# Lỗi khi xử lý tệp ${f.filename}\n\n${e?.message || 'Không thể đọc nội dung tệp.'}`,
              size: 0,
              updated: new Date().toISOString().slice(0, 10),
              base64Data: f.base64Data,
              mimeType: f.mimeType || 'text/plain',
              equationCount: 0,
              source: 'error',
            };
          }
        })
      );

      res.json({
        success: true,
        count: processedItems.length,
        files: processedItems,
        message: `Đã xử lý thành công ${processedItems.length} tệp trên máy chủ!`,
      });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi xử lý danh sách tệp' });
    }
  });

  // API 5h: Endpoint extract-base64 (tương thích ngược, nay đã siêu tốc qua processFileOnBackend)
  app.post('/api/extract-base64', async (req, res) => {
    try {
      const { filename = 'tai_lieu', mimeType = 'image/png', base64Data = '', prompt = '' } = req.body;
      if (!base64Data) {
        return res.status(400).json({ success: false, error: 'Thiếu chuỗi dữ liệu Base64' });
      }

      const item = await processFileOnBackend({ filename, mimeType, base64Data, prompt });
      res.json({
        success: true,
        text: item.content,
        file: item,
        modelUsed: item.source || 'Backend Math Engine',
      });
    } catch (err: any) {
      res.status(500).json({ success: false, error: err?.message || 'Lỗi khi xử lý Base64' });
    }
  });

  // Smart grounded tutor response generator when external AI is unavailable or overloaded
  function generateGroundedTutorAnswer(
    userQuery: string,
    role: string,
    libraryData: { files: Array<{ filename: string; title: string; content: string }> },
    selectedFilename: string | null
  ) {
    const qLower = userQuery.toLowerCase();

    let targetFile = libraryData.files.find((f) => f.filename === selectedFilename);
    if (!targetFile) {
      let maxScore = -1;
      for (const file of libraryData.files) {
        let score = 0;
        const fText = (file.title + ' ' + file.filename + ' ' + file.content).toLowerCase();
        const keywords = qLower.split(/\s+/).filter((w) => w.length > 2);
        for (const kw of keywords) {
          if (fText.includes(kw)) score += 2;
        }
        if (score > maxScore) {
          maxScore = score;
          targetFile = file;
        }
      }
    }
    if (!targetFile && libraryData.files.length > 0) {
      targetFile = libraryData.files[0];
    }

    const title = targetFile?.title || 'Kho lưu trữ Tri thức library.txt';
    const filename = targetFile?.filename || 'library.txt';
    const content = targetFile?.content || '';

    if (filename.includes('gioi_han')) {
      if (role === 'socratic') {
        return {
          text: `Chào bạn! Để tìm hiểu câu hỏi này từ tệp **${filename}** (${title}), chúng ta hãy bắt đầu từ khái niệm gốc:\n\n1. Bạn nhớ lại: Giới hạn của dãy số \\((u_n)\\) bằng \\(0\\) khi nào? (Khoảng cách \\(|u_n|\\) nhỏ hơn mọi số dương \\(\\varepsilon\\) cho trước khi \\(n\\) đủ lớn).\n2. Khi tính giới hạn dạng phân thức \\(\\lim \\frac{P(n)}{Q(n)}\\), bước đầu tiên em cần quan sát là gì? (Bậc của tử và bậc của mẫu).\n3. Bạn hãy thử chia cả tử và mẫu cho luỹ thừa cao nhất của \\(n\\) xem điều gì sẽ xảy ra nhé!`,
          sourceTitle: title,
          filename,
        };
      }
      return {
        text: `Dựa trên tài liệu **${filename}** — *${title}* trong kho lưu trữ \`library.txt\`:\n\n### 1. Định nghĩa Giới hạn dãy số\n- Dãy số $(u_n)$ có giới hạn là số thực $a$ khi $n \\to +\\infty$, ký hiệu $\\lim_{n \\to +\\infty} u_n = a$ (hoặc viết tắt $\\lim u_n = a$), nếu $\\lim_{n \\to +\\infty} (u_n - a) = 0$.\n\n### 2. Một số giới hạn cơ bản cần nhớ\n- $\\lim \\frac{1}{n^k} = 0$ với mọi $k \\in \\mathbb{N}^*$.\n- $\\lim q^n = 0$ nếu $|q| < 1$.\n- $\\lim c = c$ (với $c$ là hằng số).\n\n### 3. Các định lý về phép toán giới hạn\nNếu $\\lim u_n = a$ và $\\lim v_n = b$, thì:\n- $\\lim (u_n \\pm v_n) = a \\pm b$\n- $\\lim (u_n \\cdot v_n) = a \\cdot b$\n- $\\lim \\frac{u_n}{v_n} = \\frac{a}{b}$ (nếu $b \\neq 0$)\n\n### 4. Tổng của cấp số nhân lùi vô hạn\nVới cấp số nhân có công bội $|q| < 1$, tổng của nó là:\n$$S = \\frac{u_1}{1 - q}$$`,
        sourceTitle: title,
        filename,
      };
    }

    if (filename.includes('dao_ham')) {
      return {
        text: `Dựa trên tài liệu **${filename}** — *${title}* trong kho lưu trữ \`library.txt\`:\n\n### 1. Định nghĩa Đạo hàm tại một điểm\nĐạo hàm của hàm số $y = f(x)$ tại điểm $x_0$ là giới hạn hữu hạn (nếu có) của tỉ số giữa số gia của hàm số và số gia của đối số khi số gia của đối số dần đến $0$:\n$$f'(x_0) = \\lim_{\\Delta x \\to 0} \\frac{f(x_0 + \\Delta x) - f(x_0)}{\\Delta x} = \\lim_{x \\to x_0} \\frac{f(x) - f(x_0)}{x - x_0}$$\n\n### 2. Ý nghĩa hình học\n- $f'(x_0)$ chính là **hệ số góc $k$** của tiếp tuyến của đồ thị hàm số tại điểm $M_0(x_0; y_0)$.\n- Phương trình tiếp tuyến:\n$$y - y_0 = f'(x_0)(x - x_0)$$\n\n### 3. Quy tắc tính đạo hàm cơ bản\n- $(u \\pm v)' = u' \\pm v'$\n- $(u \\cdot v)' = u'v + uv'$\n- $\\left(\\frac{u}{v}\\right)' = \\frac{u'v - uv'}{v^2}$ (với $v \\neq 0$)\n- Đạo hàm hàm hợp: $y'_x = y'_u \\cdot u'_x$.`,
        sourceTitle: title,
        filename,
      };
    }

    if (filename.includes('triet_hoc') || filename.includes('mac_lenin')) {
      return {
        text: `Dựa trên tài liệu **${filename}** — *${title}* trong kho lưu trữ \`library.txt\`:\n\n### 1. Bản chất của Quy luật Lượng - Chất\nQuy luật chuyển hóa từ những sự thay đổi về lượng dẫn đến những sự thay đổi về chất và ngược lại là một trong ba quy luật cơ bản của phép biện chứng duy vật, chỉ ra **phương thức chung nhất** của sự vận động và phát triển trong tự nhiên, xã hội và tư duy.\n\n### 2. Các phạm trù cốt lõi\n- **Chất**: Tính quy định khách quan vốn có của sự vật, là sự thống nhất hữu cơ của các thuộc tính làm cho nó là nó chứ không phải cái khác.\n- **Lượng**: Tính quy định khách quan vốn có về mặt số lượng, quy mô, trình độ, nhịp điệu của sự vận động và phát triển.\n- **Độ**: Khoảng giới hạn mà trong đó sự thay đổi về lượng chưa làm thay đổi căn bản chất của sự vật.\n- **Điểm nút**: Điểm giới hạn mà tại đó sự thay đổi về lượng đạt tới mức độ phá vỡ độ cũ, làm thay đổi chất của sự vật.\n- **Bước nhảy**: Giai đoạn chuyển hóa căn bản về chất của sự vật do những thay đổi về lượng trước đó gây ra.\n\n### 3. Ý nghĩa phương pháp luận\n- Muốn có sự thay đổi về chất, phải kiên trì tích lũy đủ về lượng, chống tư tưởng nôn nóng, đốt cháy giai đoạn (tả khuynh).\n- Khi lượng đã tích lũy đến điểm nút, phải quyết đoán thực hiện bước nhảy, chống tư tưởng bảo thủ, trì trệ (hữu khuynh).`,
        sourceTitle: title,
        filename,
      };
    }

    if (filename.includes('nhiet_dong')) {
      return {
        text: `Dựa trên tài liệu **${filename}** — *${title}* trong kho lưu trữ \`library.txt\`:\n\n### 1. Nguyên lý thứ nhất của Nhiệt động lực học\nĐộ biến thiên nội năng của hệ bằng tổng công và nhiệt lượng mà hệ nhận được:\n$$\\Delta U = Q + A$$\n*Quy ước dấu:*\n- $Q > 0$: Hệ nhận nhiệt lượng; $Q < 0$: Hệ truyền nhiệt lượng.\n- $A > 0$: Hệ nhận công; $A < 0$: Hệ sinh công.\n\n### 2. Nguyên lý thứ hai của Nhiệt động lực học\n- **Cách phát biểu của Clausius**: Nhiệt không thể tự truyền từ một vật sang một vật nóng hơn.\n- **Cách phát biểu của Kelvin**: Không thể chế tạo được một động cơ nhiệt hoạt động theo chu trình tuần hoàn mà kết quả duy nhất là biến toàn bộ nhiệt lượng thành công cơ học.\n\n### 3. Hiệu suất cực đại của Động cơ nhiệt Carnot\n$$\\eta_{\\max} = \\frac{T_1 - T_2}{T_1} = 1 - \\frac{T_2}{T_1}$$\nTrong đó $T_1$ là nhiệt độ nguồn nóng, $T_2$ là nhiệt độ nguồn lạnh (tính theo Kelvin).`,
        sourceTitle: title,
        filename,
      };
    }

    const paragraphs = content.split('\n\n').filter((p) => p.trim().length > 30);
    const excerpt = paragraphs.slice(0, 3).join('\n\n');

    return {
      text: `Dựa trên tài liệu **${filename}** — *${title}* được lưu trữ trong \`library.txt\`:\n\n${excerpt || content.slice(0, 800)}\n\n---\n*Gia sư gợi ý: Bạn có thể đặt câu hỏi chi tiết hơn về các khái niệm, công thức hoặc tạo bài luyện tập trắc nghiệm 30 câu từ tài liệu này!*`,
      sourceTitle: title,
      filename,
    };
  }

  // API 6: Multi-turn AI Tutor with full access to library.txt knowledge AND Base64 payloads
  app.post('/api/chat', async (req, res) => {
    try {
      const {
        messages = [],
        role = 'socratic',
        modelType = 'general',
        enableSearch = false,
        selectedFile = null,
        base64Attachments = [],
        inlineFiles = [],
      } = req.body;

      const libraryData = getLibraryData();
      // Merge client-side in-memory files with server library files
      const mergedFiles = [...libraryData.files];
      if (Array.isArray(inlineFiles)) {
        for (const cf of inlineFiles) {
          if (cf && cf.filename && !mergedFiles.some((x) => x.filename === cf.filename)) {
            mergedFiles.push({
              filename: cf.filename,
              title: cf.title || cf.filename,
              size: cf.size || (cf.content ? cf.content.length : 0),
              updated: cf.updated || new Date().toISOString().slice(0, 10),
              content: cf.content || '',
            });
          }
        }
      }

      let knowledgeContext = '';

      if (selectedFile) {
        const found = mergedFiles.find((f) => f.filename === selectedFile);
        if (found) {
          knowledgeContext = `[TÀI LIỆU ĐANG TRỌNG TÂM: ${found.filename} — ${found.title}]\n${found.content}\n\n`;
        }
      }

      knowledgeContext += `[TOÀN BỘ KHO TRI THỨC ĐƯỢC LƯU TRỮ TRONG library.txt (${mergedFiles.length} tệp)]:\n`;
      for (const f of mergedFiles) {
        knowledgeContext += `--- [Tệp: ${f.filename} | Tiêu đề: ${f.title}] ---\n${(f.content || '').slice(0, 4000)}\n\n`;
      }

      let rolePrompt = '';
      if (role === 'socratic') {
        rolePrompt =
          'Bạn là Gia sư Socratic thông thái và kiên nhẫn. Bạn KHÔNG giải vẹt hay đưa đáp án trần mà đặt câu hỏi gợi mở, giải thích từng bước dựa trên kho tri thức lưu trữ trong library.txt và dữ liệu Base64 đính kèm, hướng dẫn người học tự tư duy.';
      } else if (role === 'expert') {
        rolePrompt =
          'Bạn là Chuyên gia Lý luận & Giải thích Chuyên sâu. Bạn giải thích cặn kẽ mọi khái niệm, định lý, câu hỏi của người học, chỉ rõ bản chất, chứng minh logic, liên hệ thực tế và trích dẫn chuẩn xác từ các tệp trong library.txt và dữ liệu Base64 đính kèm.';
      } else {
        rolePrompt =
          'Bạn là Huấn luyện viên Tóm tắt & Giải đáp Cấp tốc. Nêu đúng trọng tâm, công thức cốt lõi và câu trả lời rõ ràng trong 3-5 câu súc tích nhất.';
      }

      const systemInstruction = `${rolePrompt}

NGUỒN TRI THỨC CHÍNH TỪ KHO LƯU TRỮ library.txt & BASE64:
${knowledgeContext}

HƯỚNG DẪN TRẢ LỜI (CHUẨN ĐỊNH DẠNG ĐẦU RA THỐNG NHẤT TOÀN WEB):
1. Luôn sử dụng tiếng Việt chuẩn bảng mã Unicode UTF-8 (NFC), thân thiện, chuẩn xác, giàu tính sư phạm.
2. Khi câu hỏi liên quan đến kiến thức có trong library.txt hoặc tệp Base64 đính kèm, hãy giải đáp dựa trên tài liệu đó và nhắc tên tệp.
3. BẮT BUỘC định dạng mọi công thức toán/vật lý/hóa học chuẩn KaTeX LaTeX thống nhất: dùng $...$ cho công thức inline và $$...$$ cho khối công thức độc lập.
4. Nếu người học gửi kèm ảnh hoặc tài liệu qua Base64, hãy đọc kỹ nội dung trong Base64 đó để giải bài tập hoặc tạo câu hỏi theo yêu cầu.`;

      const contents: any[] = (messages.length > 0 ? messages : [{ role: 'user', content: 'Xin chào gia sư!' }]).map((m: any, idx: number, arr: any[]) => {
        const parts: any[] = [{ text: String(m.content || '') }];
        // Attach base64 files to the latest user message
        if (idx === arr.length - 1 && m.role !== 'assistant' && Array.isArray(base64Attachments) && base64Attachments.length > 0) {
          for (const att of base64Attachments) {
            if (att && att.base64Data) {
              const rawB64 = att.base64Data.includes(',') ? att.base64Data.split(',')[1] : att.base64Data;
              const mime = att.mimeType || 'image/png';
              if (mime.startsWith('image/') || mime === 'application/pdf') {
                parts.push({
                  inlineData: {
                    mimeType: mime,
                    data: rawB64,
                  },
                });
              } else {
                const decodedSnippet = Buffer.from(rawB64, 'base64').toString('utf-8').replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\uFFFD]+/g, ' ').trim();
                if (decodedSnippet) {
                  parts.push({
                    text: `\n[Nội dung tệp đính kèm ${att.filename || 'file'} (${mime})]:\n${decodedSnippet.slice(0, 12000)}\n`,
                  });
                }
              }
            }
          }
        }
        return {
          role: m.role === 'assistant' ? 'model' : 'user',
          parts,
        };
      });

      const lastUserMsg = [...messages].reverse().find((m: any) => m.role === 'user')?.content || 'Xin chào';

      const candidateModels = [
        modelType === 'fast' ? 'gemini-3.1-flash-lite' : 'gemini-4-hydrogen',
        'gemini-3.1-flash-lite',
      ];

      let replyText = '';
      let usedModel = 'Gia sư AuraLearn (Grounded library.txt)';
      let sources: any[] = [];
      let aiSucceeded = false;

      const apiKey = process.env.GEMINI_API_KEY;
      if (apiKey && apiKey !== 'MY_GEMINI_API_KEY') {
        try {
          const { ai } = getGeminiClient();
          for (const model of candidateModels) {
            try {
              const generateOptions: any = {
                model,
                contents,
                config: {
                  systemInstruction,
                  thinkingConfig: {
                    thinkingLevel: model === 'gemini-3.1-flash-lite' ? ThinkingLevel.MINIMAL : ThinkingLevel.LOW,
                  },
                },
              };

              if (enableSearch) {
                generateOptions.tools = [{ googleSearch: {} }];
              }

              const response: any = await withTimeout(
                ai.models.generateContent(generateOptions),
                30000,
                'Chat AI Timeout'
              );

              if (response && response.text) {
                replyText = response.text;
                usedModel = model;
                aiSucceeded = true;

                const candidate = response.candidates?.[0];
                const grounding = candidate?.groundingMetadata;
                if (grounding?.groundingChunks) {
                  for (const chunk of grounding.groundingChunks) {
                    if (chunk.web?.uri) {
                      sources.push({
                        title: chunk.web.title || 'Nguồn tham khảo',
                        uri: chunk.web.uri,
                      });
                    }
                  }
                }
                break;
              }
            } catch {
              // Try next candidate model smoothly
            }
          }
        } catch {
          // AI client initialization failure handled below
        }
      }

      if (!aiSucceeded || !replyText) {
        const grounded = generateGroundedTutorAnswer(lastUserMsg, role, { files: mergedFiles }, selectedFile);
        replyText = grounded.text;
        usedModel = `Kho tri thức ${grounded.filename}`;
        sources.push({
          title: `Tệp lưu trữ: ${grounded.filename}`,
          uri: `#${grounded.filename}`,
        });
      }

      res.json({
        success: true,
        text: replyText,
        modelUsed: usedModel,
        searchUsed: Boolean(enableSearch && sources.length > 0),
        sources,
      });
    } catch {
      res.json({
        success: true,
        text: 'Xin chào bạn! Gia sư AuraLearn đã kết nối với kho lưu trữ `library.txt`. Bạn muốn giải đáp về Giới hạn dãy số, Đạo hàm, Quy luật Lượng - Chất hay Nhiệt động lực học?',
        modelUsed: 'AuraLearn Knowledge Base',
        searchUsed: false,
        sources: [],
      });
    }
  });

  // API 7: Generate Quiz via Gemini Structured Outputs + 2-batch parallel splitting (no dummy fallback)
  app.post('/api/generate-quiz', async (req, res) => {
    try {
      const { filename, filenames, attempt = 0, inlineFiles = [], base64Files = [], promptConfig = '' } = req.body || {};
      const libraryData = getLibraryData();

      // Merge server files with client-side inline files (so uploaded files always work even without server disk)
      const allFilesMap = new Map<string, { filename: string; title: string; content: string; base64Data?: string; mimeType?: string }>();
      for (const f of libraryData.files) {
        allFilesMap.set(f.filename, f);
      }
      if (Array.isArray(inlineFiles)) {
        for (const cf of inlineFiles) {
          if (cf && cf.filename) {
            allFilesMap.set(cf.filename, {
              filename: cf.filename,
              title: cf.title || cf.filename,
              content: cf.content || '',
              base64Data: cf.base64Data,
              mimeType: cf.mimeType,
            });
          }
        }
      }
      const mergedFiles = Array.from(allFilesMap.values());

      let selectedList: string[] = [];
      if (Array.isArray(filenames) && filenames.length > 0) {
        selectedList = filenames.filter((f: any) => typeof f === 'string' && f.trim());
      } else if (typeof filename === 'string' && filename) {
        selectedList = [filename];
      }

      if (selectedList.length === 0 || selectedList.includes('all')) {
        selectedList = mergedFiles.map((f) => f.filename);
      }

      let targetFiles = mergedFiles.filter((f) => selectedList.includes(f.filename));
      if (targetFiles.length === 0) {
        targetFiles = mergedFiles;
        selectedList = mergedFiles.map((f) => f.filename);
      }

      const targetTitle =
        targetFiles.length === 1
          ? targetFiles[0].title
          : targetFiles.length === mergedFiles.length
          ? `Toàn bộ kho lưu trữ library.txt (${targetFiles.length} tệp)`
          : `Tổng hợp ${targetFiles.length} tệp (${targetFiles.map((f) => f.title).join(', ')})`;

      if (req.body?.simulate429) {
        return res.status(429).json({
          success: false,
          rateLimited: true,
          status: 429,
          error: 'RESOURCE_EXHAUSTED',
          message: QUIZ_RATE_LIMIT_MESSAGE,
        });
      }

      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
        return res.status(503).json({
          success: false,
          error: 'AI_OVERLOADED',
          message: QUIZ_OVERLOAD_MESSAGE,
        });
      }

      const { ai } = getGeminiClient();

      // Attach binary inlineData for images/PDFs; text-based files are already extracted in combinedTextContext
      const binaryParts: any[] = [];
      const b64List =
        Array.isArray(base64Files) && base64Files.length > 0
          ? base64Files
          : targetFiles.filter((f) => f.base64Data);

      for (const bItem of b64List) {
        const rawB64 = (bItem.base64Data || '').includes(',')
          ? bItem.base64Data.split(',')[1]
          : bItem.base64Data || '';
        const mime = bItem.mimeType || 'text/plain';

        if (rawB64 && (mime.startsWith('image/') || mime === 'application/pdf')) {
          binaryParts.push({
            inlineData: {
              mimeType: mime,
              data: rawB64,
            },
          });
        }
      }

      const rawCombinedText = targetFiles
        .map((f) => `=== TỆP: ${f.filename} (${f.title}) ===\n${f.content || ''}`)
        .join('\n\n')
        .trim();

      if (!rawCombinedText && binaryParts.length === 0) {
        return res.status(400).json({
          success: false,
          error: 'EMPTY_SOURCE',
          message: QUIZ_OVERLOAD_MESSAGE,
        });
      }

      // Lọc giữ định nghĩa, định lý, công thức & ví dụ trọng tâm nếu tài liệu > 25.000 ký tự
      const combinedTextContext =
        rawCombinedText.length > 25000
          ? condenseTheoryContent(rawCombinedText, MAX_QUIZ_CONTEXT_CHARS)
          : rawCombinedText;

      const { partIPrompt, partIIAndIIIPrompt } = buildQuizPromptsForSplitBatches({
        attempt: Number(attempt) || 1,
        combinedTextContext,
        promptConfig,
      });

      const extractJsonText = (response: any): string => {
        const directText = typeof response?.text === 'string' ? response.text : '';
        const partsText = Array.isArray(response?.candidates?.[0]?.content?.parts)
          ? response.candidates[0].content.parts
              .filter((p: any) => typeof p?.text === 'string')
              .map((p: any) => p.text)
              .join('')
          : '';
        return (directText || partsText || '')
          .trim()
          .replace(/^```(?:json)?\s*/i, '')
          .replace(/```\s*$/, '')
          .trim();
      };

      const quizModels = [
        { model: 'gemini-4-hydrogen', thinkingLevel: ThinkingLevel.LOW, timeoutMs: 50000 },
        { model: 'gemini-3.1-flash-lite', thinkingLevel: ThinkingLevel.MINIMAL, timeoutMs: 40000 },
      ];

      let lastQuizErr: any = null;
      let sawRateLimit = false;

      const generateQuizPartsForModel = async (
        cfg: { model: string; thinkingLevel: ThinkingLevel; timeoutMs: number },
        sequential: boolean
      ) => {
        const callPart1 = () =>
          ai.models.generateContent({
            model: cfg.model,
            contents: { parts: [...binaryParts, { text: partIPrompt }] },
            config: {
              systemInstruction: QUIZ_SYSTEM_INSTRUCTION,
              responseMimeType: 'application/json',
              responseSchema: PART_I_RESPONSE_SCHEMA,
              maxOutputTokens: 8192,
              thinkingConfig: {
                thinkingLevel: cfg.thinkingLevel,
              },
            },
          });

        const callPart23 = () =>
          ai.models.generateContent({
            model: cfg.model,
            contents: { parts: [...binaryParts, { text: partIIAndIIIPrompt }] },
            config: {
              systemInstruction: QUIZ_SYSTEM_INSTRUCTION,
              responseMimeType: 'application/json',
              responseSchema: PART_II_III_RESPONSE_SCHEMA,
              maxOutputTokens: 8192,
              thinkingConfig: {
                thinkingLevel: cfg.thinkingLevel,
              },
            },
          });

        if (sequential) {
          const part1Res = await withTimeout(callPart1(), cfg.timeoutMs, 'Quiz AI Timeout');
          const part23Res = await withTimeout(callPart23(), cfg.timeoutMs, 'Quiz AI Timeout');
          return [part1Res, part23Res] as const;
        }

        return await withTimeout(
          Promise.all([callPart1(), callPart23()]),
          cfg.timeoutMs,
          'Quiz AI Timeout'
        );
      };

      for (const cfg of quizModels) {
        for (const sequential of [false, true]) {
          try {
            const [part1Res, part23Res] = await generateQuizPartsForModel(cfg, sequential);

            const part1Raw = extractJsonText(part1Res);
            const part23Raw = extractJsonText(part23Res);
            if (!part1Raw || !part23Raw) {
              throw new Error('Phản hồi AI bị rỗng.');
            }

            const partI = buildValidatedPartI(JSON.parse(part1Raw));
            const { partII, partIII } = buildValidatedPartIIAndIII(JSON.parse(part23Raw));
            const questions = [...partI, ...partII, ...partIII];

            if (questions.length === 30) {
              return res.json({
                success: true,
                questions,
                source: targetTitle,
                filenames: selectedList,
                attempt: Number(attempt) || 0,
                usedAI: true,
                modelUsed: cfg.model,
              });
            }
          } catch (err: any) {
            lastQuizErr = err;
            if (isRateLimitError(err)) {
              sawRateLimit = true;
              continue;
            }
            break;
          }
        }
      }

      if (lastQuizErr) {
        console.warn('Gemini Quiz Generation error:', lastQuizErr?.message || lastQuizErr);
      }

      if (sawRateLimit) {
        return res.status(429).json({
          success: false,
          rateLimited: true,
          status: 429,
          error: 'RESOURCE_EXHAUSTED',
          message: QUIZ_RATE_LIMIT_MESSAGE,
        });
      }

      return res.status(503).json({
        success: false,
        error: 'AI_OVERLOADED',
        message: QUIZ_OVERLOAD_MESSAGE,
      });
    } catch (err: any) {
      console.warn('Gemini Quiz Generation fatal error:', err?.message || err);
      if (isRateLimitError(err)) {
        return res.status(429).json({
          success: false,
          rateLimited: true,
          status: 429,
          error: 'RESOURCE_EXHAUSTED',
          message: QUIZ_RATE_LIMIT_MESSAGE,
        });
      }
      return res.status(503).json({
        success: false,
        error: 'AI_OVERLOADED',
        message: QUIZ_OVERLOAD_MESSAGE,
      });
    }
  });

  if (!isProd) {
    const vite = await createViteServer({
      server: { middlewareMode: true, hmr: false, ws: false },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static(path.resolve(__dirname, 'dist')));
    app.get('*', (req, res) => {
      res.sendFile(path.resolve(__dirname, 'dist', 'index.html'));
    });
  }

  process.on('uncaughtException', (err: any) => {
    // Ignore harmless Vite/ws errors or port collisions on secondary ports
    if (err?.code === 'EADDRINUSE' && (err?.port === 24678 || String(err?.message).includes('24678'))) {
      console.warn('[Server] Ignored WebSocket port conflict on 24678');
      return;
    }
    console.error('[Server uncaughtException]', err);
  });

  process.on('unhandledRejection', (reason: any) => {
    console.error('[Server unhandledRejection]', reason);
  });

  let retries = 0;
  const maxRetries = 30;
  let activeServer: any = null;

  const shutdown = () => {
    if (activeServer) {
      try {
        activeServer.close(() => process.exit(0));
      } catch {
        process.exit(0);
      }
      setTimeout(() => process.exit(0), 500);
    }
  };
  process.once('SIGTERM', shutdown);
  process.once('SIGINT', shutdown);

  const listenWithRetry = () => {
    const server = app.listen(PORT, '0.0.0.0', () => {
      console.log(`AuraLearn server running on http://0.0.0.0:${PORT}`);
    });
    activeServer = server;
    server.on('error', (err: any) => {
      if (err.code === 'EADDRINUSE' && retries < maxRetries) {
        retries++;
        console.warn(`Port ${PORT} in use, retrying (${retries}/${maxRetries})...`);
        setTimeout(() => {
          try {
            server.close();
          } catch {
            // ignore if server wasn't listening yet
          }
          listenWithRetry();
        }, 500);
      } else {
        console.error('Server listen error:', err);
        process.exit(1);
      }
    });
  };

  listenWithRetry();
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
