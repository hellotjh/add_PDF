import os
import json
import tempfile
import zipfile
from io import BytesIO
from xml.etree import ElementTree as ET

from langchain_classic.chains import ConversationChain, ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import pandas as pd


def _decode_text(file_bytes):
    """把上传文件的二进制内容解码成字符串。

    不同来源的文本文件编码不一定相同。中文 Windows 环境里常见 GBK/GB18030，
    现代编辑器常见 UTF-8，所以这里按常见编码依次尝试。
    """
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="ignore")


def _docx_bytes_to_text(file_bytes):
    """从 docx 文件中提取正文文本。

    docx 本质上是一个 zip 压缩包，正文 XML 通常在 word/document.xml。
    这里用标准库读取 XML，不额外安装 python-docx 依赖。
    """
    # 打开 docx 压缩包，并读取正文 XML。
    with zipfile.ZipFile(BytesIO(file_bytes)) as archive:
        xml_bytes = archive.read("word/document.xml")

    # 解析 WordprocessingML，w:t 节点里保存了实际文本片段。
    root = ET.fromstring(xml_bytes)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []

    # 以段落 w:p 为单位拼接文本，这样能尽量保留 Word 原来的段落结构。
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text for node in paragraph.findall(".//w:t", namespace) if node.text]
        line = "".join(texts).strip()
        if line:
            paragraphs.append(line)

    return "\n".join(paragraphs)


def _load_uploaded_documents(upload_file):
    """根据上传文件后缀，把文件内容转换成 LangChain Document 列表。

    后面的向量库构建逻辑只认识 Document，所以不同文件格式都在这里统一转换。
    """
    # Streamlit UploadedFile 通常有 name 属性，扩展名用于判断加载方式。
    filename = getattr(upload_file, "name", "")
    suffix = os.path.splitext(filename)[1].lower()

    # 上传文件可能已经被读取过，先把读取位置重置到开头。
    if hasattr(upload_file, "seek"):
        upload_file.seek(0)
    file_bytes = upload_file.read()

    if suffix == ".pdf":
        # PyPDFLoader 只能接收文件路径，不能直接接收 bytes。
        # 所以这里先写入系统临时 PDF，再交给 PyPDFLoader 读取。
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as file:
                file.write(file_bytes)
                temp_file_path = file.name

            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            if not docs:
                raise ValueError("PDF 中没有读取到页面内容，请确认上传的是有效 PDF。")
            return docs
        finally:
            # 无论读取成功还是失败，都清理临时文件。
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    if suffix in {".txt", ".md", ".markdown", ".py", ".js", ".ts", ".html", ".htm"}:
        # 纯文本类文件直接解码成字符串，然后封装成一个 Document。
        text = _decode_text(file_bytes).strip()
        if not text:
            raise ValueError("文本文件为空，无法进行问答。")
        return [Document(page_content=text, metadata={"source": filename, "file_type": suffix.lstrip(".")})]

    if suffix == ".json":
        # JSON 优先格式化成缩进文本，方便模型理解层级结构。
        text = _decode_text(file_bytes).strip()
        if not text:
            raise ValueError("JSON 文件为空，无法进行问答。")
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False, indent=2)
        except json.JSONDecodeError:
            # 如果不是标准 JSON，就退回普通文本处理。
            pass
        return [Document(page_content=text, metadata={"source": filename, "file_type": "json"})]

    if suffix == ".csv":
        # CSV 使用 pandas 读取后再转回标准 CSV 文本，能处理常见表格格式问题。
        df = pd.read_csv(BytesIO(file_bytes))
        text = df.fillna("").to_csv(index=False).strip()
        if not text:
            raise ValueError("CSV 文件为空，无法进行问答。")
        return [Document(page_content=text, metadata={"source": filename, "file_type": "csv"})]

    if suffix == ".xlsx":
        # Excel 可能有多个 sheet，这里把每个 sheet 都转成文本并保留 sheet 名。
        sheets = pd.read_excel(BytesIO(file_bytes), sheet_name=None)
        parts = []
        for sheet_name, df in sheets.items():
            sheet_text = df.fillna("").to_csv(index=False).strip()
            if sheet_text:
                parts.append(f"[Sheet: {sheet_name}]\n{sheet_text}")
        if not parts:
            raise ValueError("Excel 文件中没有可读取的数据。")
        return [Document(page_content="\n\n".join(parts), metadata={"source": filename, "file_type": "xlsx"})]

    if suffix == ".docx":
        # docx 走标准库 XML 解析，提取正文段落。
        text = _docx_bytes_to_text(file_bytes).strip()
        if not text:
            raise ValueError("DOCX 文件中没有提取到文本。")
        return [Document(page_content=text, metadata={"source": filename, "file_type": "docx"})]

    raise ValueError("暂不支持该文件类型，请上传 pdf、txt、md、csv、json、docx 或 xlsx 文件。")


def qa_agent(open_api_key, memory, upload_file, question):
    """基于上传文件进行检索增强问答。

    参数说明：
    - open_api_key：聊天模型使用的 API key。
    - memory：LangChain 对话记忆，用于保存历史问答。
    - upload_file：Streamlit 上传的文件对象。
    - question：用户当前问题。
    """
    # 优先使用函数传入的密钥，未传入时再读取环境变量。
    api_key = open_api_key or os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI-API-KEY")
    base_url = "https://xcode.best/v1"

    # 初始化对话模型。
    model = ChatOpenAI(
        model="gpt-5.5",
        api_key=api_key,
        base_url=base_url,
    )

    # 根据文件类型加载文档。
    docs = _load_uploaded_documents(upload_file)

    # 按中文标点和换行切分文档，保留少量重叠以减少上下文断裂。
    # texts 是 Document 切片列表，后面会被送进向量库。
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n\n", "\n\n", "\n", "。", "！", "？", "，", "、", "...", ""],
    )
    texts = text_splitter.split_documents(docs)
    if not texts:
        raise ValueError("文件中没有提取到可检索文本。")

    # 生成向量索引，并把 FAISS 包装成检索器。
    # 用户提问时，retriever 会从这些切片里找出最相关的内容交给模型。
    embeddings_model = OpenAIEmbeddings(
        model="text-embedding-v4",
        api_key="sk-1ef2d61d64764c6292defdadfe65d500",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        dimensions=1024,
        check_embedding_ctx_length=False,
        chunk_size=10
    )
    db = FAISS.from_documents(texts, embeddings_model)
    retriever = db.as_retriever()

    # 使用带记忆的检索问答链回答当前问题。
    # qa.invoke 会自动读取 memory 中的历史对话，并在回答后写回新的问答。
    qa = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        memory=memory,
        chain_type="map_reduce"
    )
    response = qa.invoke({"question": question})
    return response


def qb_agent(openai_key,memory,question):
    """没有上传文件时使用的普通聊天代理。"""
    model=ChatOpenAI(model="gpt-5.5",base_url="https://xcode.best/v1",
                     api_key=openai_key)

    # 当前项目的 memory_key 是 chat_history，因此普通聊天链也要使用同名变量。
    # MessagesPlaceholder 会把历史消息列表原样插入到聊天模型上下文里。
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个有帮助的 AI 助手。"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])

    chain=ConversationChain(
        llm=model,
        memory=memory,
        prompt=prompt,
        output_key="answer",
    )
    response=chain.invoke({"input":question})
    return response
