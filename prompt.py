from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from flask_cors import CORS
from langchain.chains import LLMChain
from langchain.chains import retrieval_qa
from langchain_core.messages import SystemMessage
from langchain.prompts import MessagesPlaceholder,HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from flask import Flask, request, jsonify

load_dotenv()

app = Flask(__name__)
CORS(app)


chat = ChatOllama(model="llama3")

embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

db = Chroma(
    embedding_function=embeddings,
    persist_directory="emb"
)
prompt = ChatPromptTemplate(
    input_variables=["query"],
    messages=[
        SystemMessage(content="You're a chatbot that is integrated in a online portfolio website of a person named Srilakshman. You're to use the context given and answer the questions of the user who want to know about Srilakshman. The context is information regarding Srilakshman."),
        HumanMessagePromptTemplate.from_template(
            "Answer the question based only on the following context(But dont mention that you are answering based on context): {context} \n--\nAnswer the question based on the above context: {query}"
        )
    ]
)
chain = LLMChain(
    llm=chat,
    prompt=prompt,
    output_parser=StrOutputParser(),
)

@app.route('/chat', methods=['POST'])
def chat():
    if request.method == 'POST':
        data = request.json
        query = data['query']
        results = db.similarity_search_with_score(query, k = 2)
        context = "\n\n-------\n\n".join([doc.page_content for doc, _score in results])
        result = chain.invoke({"context":context, "query":query})
        print(result)
        return jsonify({'response': result['text']})
    
app.run(host='0.0.0.0', port=5550, debug=True)