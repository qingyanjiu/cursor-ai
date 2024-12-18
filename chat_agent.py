from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List

def create_chat_agent():
    # 初始化Ollama聊天模型
    chat_model = ChatOllama(
        model="llama3-cn",
        temperature=0.2,
        base_url="http://localhost:11434",
        streaming=False,  # 启用流式输出
        seed=42,  # 添加随机种子以保持一致性
    )
    return chat_model

def chat():
    chat_model = create_chat_agent()
    messages: List = [
        SystemMessage(content="你是一个有帮助的AI助手。请用中文回答问题。")
    ]
    
    print("开始对话 (输入 'quit' 退出)")
    
    while True:
        user_input = input("\n用户: ")
        if user_input.lower() == 'quit':
            break
        
        # 添加用户消息
        messages.append(HumanMessage(content=user_input))
        
        try:
            # 获取AI响应
            response = chat_model.invoke(messages)
            print(f"\nAI: {response.content}")
            
            # 添加AI响应到历史记录
            messages.append(response)
            
            # 保持历史记录在合理范围内
            if len(messages) > 8:  # 保留系统消息和最近的3轮对话
                messages = [messages[0]] + messages[-6:]
        except Exception as e:
            print(f"\n发生错误: {e}")
            continue

if __name__ == "__main__":
    chat() 