from langchain.chains.conversation.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain, ConversationalRetrievalChain
from langchain import PromptTemplate, LLMChain, HuggingFaceHub
# from langchain.llms import CTransformers
import os
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceBgeEmbeddings
# from io import BytesIO
from langchain.document_loaders import PyPDFLoader
import random
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chains import ConversationChain, ConversationalRetrievalChain


from django.shortcuts import render, redirect
from django.http import JsonResponse
import openai

from django.contrib import auth
from django.contrib.auth.models import User
from .models import Chat

from django.utils import timezone


model_name = "BAAI/bge-large-en"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
embeddings = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

print("Embeddings model loaded....")


token = "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"

llm=HuggingFaceHub(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1", model_kwargs={"temperature":0.1, "max_length":512}, huggingfacehub_api_token = token)
print("LLM Initialized...")


prompt_template = """Use the following pieces of information to answer the user's question related to Substation Asset Maintenance.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and Frame it grammatically correct.
Helpful answer:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
load_vector_store = Chroma(persist_directory="stores/substation_cosine", embedding_function=embeddings)

retriever = load_vector_store.as_retriever(search_kwargs={"k":1})


memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')

chain_type_kwargs = {"prompt": prompt}

conv_qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    get_chat_history=lambda h : h,
    return_source_documents=True)

def grammatical(text):
    paragraphs = text.split('.')
    last_paragraph = paragraphs[-1]
    if '.' not in paragraphs[-1]:
        truncated_text = '.'.join(paragraphs[:-1]) + '.'
    else:
        truncated_text = text 
    return truncated_text

import pyttsx3

def get_response(message):
    query = message
    chain_type_kwargs = {"prompt": prompt}
    conv_qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        get_chat_history=lambda h: h,
        return_source_documents=True)
    response = conv_qa_chain(query)
    answer_text = grammatical(response['answer'])
    
    return answer_text


#audio_file_path = text_to_audio(text_input)

# Create your views here.

def index(request):
    return render(request, 'index.html')

def history(request):
    chats = Chat.objects.filter(user=request.user)
    return render(request, 'history.html', {'chats': chats})

def newchat(request):
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
      message = request.POST.get('message')
      response = get_response(message)

      chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
      chat.save()
      

      return JsonResponse({'message': message, 'response': response})

    elif request.method == 'POST' and 'chat_id' in request.POST and 'feedback_type' in request.POST:
      chat_id = request.POST.get('chat_id')
      feedback_type = request.POST.get('feedback_type')

      chat = Chat.objects.get(id=chat_id)

      if feedback_type == 'like':
         chat.like = True
         chat.dislike = False
      elif feedback_type == 'dislike':
         chat.like = False
         chat.dislike = True

      chat.save()

      return JsonResponse({'success': True})

    return render(request, 'newchat.html', {'chats': chats})

def speak_text(answer_text):
    engine = pyttsx3.init()
    engine.say(answer_text)
    engine.runAndWait()


def chatbot(request):
    chats = Chat.objects.filter(user=request.user)

    if request.method == 'POST':
      message = request.POST.get('message')
      response = get_response(message)

      chat = Chat(user=request.user, message=message, response=response, created_at=timezone.now())
      chat.save()

      return JsonResponse({'message': message, 'response': response})

    elif request.method == 'POST' and 'chat_id' in request.POST and 'feedback_type' in request.POST:
      chat_id = request.POST.get('chat_id')
      feedback_type = request.POST.get('feedback_type')

      chat = Chat.objects.get(id=chat_id)

      if feedback_type == 'like':
         chat.like = True
         chat.dislike = False
      elif feedback_type == 'dislike':
         chat.like = False
         chat.dislike = True

      chat.save()

      return JsonResponse({'success': True})

    return render(request, 'chatbot.html', {'chats': chats})


def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = auth.authenticate(request, username=username, password=password)
        if user is not None:
            auth.login(request, user)
            return redirect('chatbot')
        else:
            error_message = 'Invalid Username or Password'
            return render(request, 'login.html', {'error_message': error_message})
    else:
        return render(request, 'login.html')


def register(request):
    if request.method == 'POST':
        username = request.POST['username']
        email = request.POST['email']
        password1 = request.POST['password1']
        password2 = request.POST['password2']

        if password1 == password2:
            try:
                user = User.objects.create_user(username, email, password1)
                user.save()
                auth.login(request, user)
                return redirect('login')
            except:
                error_message = 'Error creating account'
                return render(request, 'register.html', {'error_message': error_message})
        else:
            error_message = 'Password not Match'
            return render(request, 'register.html', {'error_message': error_message})
    return render(request, 'register.html')


def logout(request):
    auth.logout(request)
    return redirect('login')
