import streamlit as st
import tempfile
import os

# Importaciones de LangChain
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# 1. Configuración de la Página
st.set_page_config(page_title="Conchita RAG Explorer", page_icon="🕵️‍♀️", layout="wide")
st.title("🕵️‍♀️ Conchita RAG: Preguntas con Trazabilidad")
st.markdown("Sube un documento (PDF/TXT), pregunta y **mira qué está leyendo la IA**.")

# 2. Sidebar: Configuración y Carga
with st.sidebar:
    st.header("1. Configuración")
    api_key = st.text_input("Google API Key:", type="password")

    st.header("2. Tus Datos")
    uploaded_file = st.file_uploader("Sube tu archivo", type=["pdf", "txt"])

    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key

# 3. Función de Procesamiento
@st.cache_resource
def procesar_documento(uploaded_file):
    # Guardar archivo temporalmente
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    # A. Loader
    if uploaded_file.name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path)

    docs = loader.load()

    # B. Splitter (Segmentación)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)

    # C. Embedding y VectorStore (Chroma)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

    return vectorstore.as_retriever()

# 4. Lógica Principal de la App
if uploaded_file and api_key:
    with st.spinner("Procesando documento e indexando vectores..."):
        try:
            retriever = procesar_documento(uploaded_file)
            st.success("✅ Documento indexado en ChromaDB")
        except Exception as e:
            st.error(f"Error al procesar: {e}")
            st.stop()

    user_question = st.text_input("Pregunta algo sobre tu documento:")

    if user_question:
        # Paso 1: Recuperar contexto
        relevant_docs = retriever.invoke(user_question)

        # Paso 2: Mostrar Trazabilidad
        with st.expander("🔍 Trazabilidad: ¿Qué fragmentos leyó la IA?"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Fragmento {i+1}** (Página {doc.metadata.get('page', 'N/A')}):")
                st.info(doc.page_content)

        # Paso 3: Generar Respuesta (Gemini)
        context_text = "\n\n".join([d.page_content for d in relevant_docs])

        template = """Responde a la pregunta basándote SOLAMENTE en el siguiente contexto:
        {context}

        Pregunta: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") # Actualizado a la versión estable

        chain = prompt | llm | StrOutputParser()

        with st.spinner("Generando respuesta..."):
            response = chain.invoke({"context": context_text, "question": user_question})
            st.subheader("💡 Respuesta:")
            st.write(response)

elif not api_key:
    st.warning("👈 Por favor introduce tu API Key en la barra lateral.")
elif not uploaded_file:
    st.info("👈 Sube un documento para comenzar.")
