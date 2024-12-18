from flask import Flask, request, jsonify, Response
from novel_chat_agent import NovelChatAgent, load_novel

app = Flask(__name__)

# 加载小说文本
novel_text = load_novel("/Users/louisliu/Downloads/test/1.docx")  # 替换为你的小说文件路径
agent = NovelChatAgent(novel_text)

@app.route('/ask', methods=['POST'])
def ask_agent():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    # 获取回答
    answer = agent.ask(question)['response']
    return jsonify({"answer": answer})

@app.route('/stream_ask', methods=['POST'])
def stream_ask_agent():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "问题不能为空"}), 400

    def generate():
        # 这里可以实现流式回答的逻辑
        # 例如，逐步返回回答的每一部分
        response = agent.ask(question)['response']  # 这里可以替换为流式处理的逻辑
        yield f"data: {response}\n\n"

    return Response(generate(), mimetype='text/event-stream')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002) 