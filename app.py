from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import os
import uuid

app = Flask(__name__)

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 질문과 답변 데이터 설정
questions = [
    "포트폴리오 주제는 뭔가요?",
    "어떤 모델을 사용했나요?",
    "프로젝트 참여 인원은 몇 명인가요?",
    "프로젝트 작업 기간은 어느 정도 인가요?",
    "조장이 누구인가요?",
    "데이터는 무얼 이용 했나요?",
    "프로젝트 하는 데 어려움은 없었나요?" ]

answers = [
    "포트폴리오 주제는 사용자 로그 학습 기반 추천 알고리즘 입니다.",
    "모델로는 NCF를 사용했습니다",
    "참여 인원은 총 3명 입니다.",
    "작업 기간은 3주 입니다.",
    "박찬혁 입니다.",
    "케글에서 영화 데이터를 다운 받아 이용했습니다.",
    "사용자 데이터를 구하기 어려웠습니다. 그래서 직접 생성하여 사용하였습니다." ]

# 질문 임베딩 생성
question_embeddings = encoder.encode(questions)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.json['user_input']
    embedding = encoder.encode([user_input])

    # 유사도 계산
    distances = cosine_similarity(embedding, question_embeddings).flatten()
    best_match_idx = distances.argmax()
    answer = answers[best_match_idx]

    # gTTS로 음성 생성
    tts = gTTS(text=answer, lang='ko')
    audio_filename = f"static/{uuid.uuid4().hex}.mp3"
    tts.save(audio_filename)

    return jsonify({"answer": answer, "audio_file": audio_filename})

if __name__ == '__main__':
    app.run(debug=True)