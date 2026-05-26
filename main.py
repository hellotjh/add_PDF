import streamlit as st
from langchain_classic.memory import ConversationBufferMemory

from utils import qb_agent, qa_agent

st.title("AI智能文件问答工具")

# 侧边栏用于输入模型 API key，type="password" 会隐藏明文。
with st.sidebar:
    openai_api_key = st.text_input("请输入OpenAI API密钥",type="password")
    st.markdown("[获取OpenAI API key](https://xcode.best)")

# Streamlit 每次交互都会重新执行脚本。
# 把 memory 放到 session_state 里，可以让同一个用户会话持续保留历史对话。
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        # return_messages=True 表示历史记录以 HumanMessage/AIMessage 对象列表保存。
        return_messages=True,
        # memory_key 必须和 qa_agent/qb_agent 里的 prompt 变量名保持一致。
        memory_key="chat_history",
        # output_key 告诉 memory：链输出里的 answer 字段就是 AI 回答。
        output_key="answer"
    )

# 支持多种文件格式；上传文件后走 qa_agent，没有上传文件时走普通聊天 qb_agent。
uploaded_file = st.file_uploader(
    "上传你的文件",
    type=["pdf", "txt", "md", "csv", "json", "docx", "xlsx"],
)
question = st.text_input("对你的文件进行提问")  # disabled=not uploaded_file
run = st.button("运行")

# 点击运行但没有 API key 时，提前停止，避免后面模型调用直接报错。
if question and not openai_api_key and run:
    st.info("请输入你的OpenAI-API-Key！")
    st.stop()

# 有上传文件时：先解析文件，再检索文件内容回答问题。
if uploaded_file and question and openai_api_key and run:
    try:
        with st.spinner("AI正在思考中，请稍等。。。"):
            response = qa_agent(openai_api_key, st.session_state["memory"], uploaded_file, question)
            st.write("### 答案")
            st.write(response["answer"])
            # qa_agent 内部的 qa.invoke 已经把本轮问答写入 memory。
            # 这里把 memory 里的消息取出来，给页面下方的历史消息区域展示。
            st.session_state["chat_history"] = st.session_state["memory"].chat_memory.messages
    except Exception as e:
        # 文件解析、向量化、模型调用等错误都会显示成页面提示，避免直接抛出长 traceback。
        st.error(str(e))

# 没有上传文件时：退化成普通聊天模式，仍然使用同一个 memory 保存上下文。
if not uploaded_file and question and openai_api_key and run:
    try:
        with st.spinner("AI正在思考中，请稍等。。。"):
            response = qb_agent(openai_api_key, st.session_state["memory"], question)
            st.write("### AI回复")
            st.write(response["answer"])
            st.session_state["chat_history"] = st.session_state["memory"].chat_memory.messages
    except Exception as e:
        st.error(str(e))

# 展示历史消息。memory 中的消息通常按“用户、AI、用户、AI”的顺序排列。
if "chat_history" in st.session_state:
    with st.expander("历史消息"):
        # 步长为 2，表示每次取一组：用户消息 + AI 消息。
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            if i + 1 >= len(st.session_state["chat_history"]):
                # 防止消息数量为奇数时，i+1 越界。
                break
            ai_message = st.session_state["chat_history"][i+1]
            st.write(f"用户提问：{human_message.content}")
            st.write(f"AI回复：{ai_message.content}")
            if i<len(st.session_state["chat_history"])-2:
                st.divider()
