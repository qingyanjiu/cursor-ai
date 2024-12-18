import os
from langchain_ollama import ChatOllama
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate

class NovelChatAgent:
    def __init__(self, novel_text, chunk_size=512):
        self.novel_text = novel_text
        self.memory = ConversationBufferMemory()
        
        # 初始化中文词嵌入模型
        self.embeddings = SentenceTransformerEmbeddings(model_name="hfl/chinese-macbert-base")  # 使用中文模型
        
        # 初始化Chroma向量数据库
        self.vector_store = Chroma(embedding_function=self.embeddings)
        
        # 将小说文本切分为多个片段并添加到向量数据库
        self.add_texts_to_vector_store(novel_text, chunk_size)
        
        # 初始化Ollama聊天模型
        self.llm = ChatOllama(
            model="llama3-cn",  # 替换为你使用的Ollama模型
            temperature=0.2,
            base_url="http://localhost:11434",  # 确保Ollama服务在此地址运行
            streaming=False,
            seed=42,
        )
    
        
        self.chain = ConversationChain(llm=self.llm, memory=self.memory)

    def add_texts_to_vector_store(self, text, chunk_size):
        # 将文本切分为多个片段
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        # 确保每个片段都是字符串
        self.vector_store.add_texts([chunk.replace("\n", " ") for chunk in chunks])

    def ask(self, question):
        
        # 在向量数据库中查找最相似的文本片段
        similar_texts = self.vector_store.similarity_search(question, k=3)  # 获取最相似的3个片段
        
        # 确保similar_texts是一个有效的列表
        if not similar_texts:
            return "没有找到相关内容。"

        # 将找到的文本片段合并为上下文
        context = "\n".join([doc.page_content.replace("\n", " ") for doc in similar_texts])  # 提取文本内容并处理换行


        # 定义PromptTemplate
        prompt = PromptTemplate.from_template("根据以下内容回答问题：\n\n{context}\n\n问题：{question}\n\n回答：")
        
        # 调整输入格式
        response = self.chain.invoke(prompt.format(context=context, question=question))
        return response

def load_novel(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

if __name__ == "__main__":
    # 设置代理
    os.environ["HTTP_PROXY"] = "http://localhost:1087"  # 设置代理地址
    os.environ["HTTPS_PROXY"] = "http://localhost:1087"  # 设置HTTPS代理地址

    novel_text = load_novel("/Users/louisliu/Downloads/test/1.txt")  # 替换为你的小说文件路径
    agent = NovelChatAgent(novel_text)
    while True:
        user_input = input("你想问什么？")
        answer = agent.ask(user_input)
        print("回答:", answer) 