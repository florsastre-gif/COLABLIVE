import streamlit as st
import os
from typing import Annotated, TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, END

# 1. Configuración de la Página
st.set_page_config(page_title="IA para Jóvenes", page_icon="📖", layout="wide")
st.title("📖 El Cuentacuentos de la IA")
st.markdown("Pregunta sobre cualquier novedad de IA y te la explicaré como una historia para jóvenes.")

# 2. Sidebar: Configuración de API Keys
with st.sidebar:
    st.header("🔑 Configuración")
    google_key = st.text_input("Google API Key:", type="password")
    tavily_key = st.text_input("Tavily API Key:", type="password")
    
    if google_key and tavily_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["TAVILY_API_KEY"] = tavily_key
        st.success("APIs configuradas")

# 3. Estructura de LangGraph (El Cerebro)
class AgentState(TypedDict):
    question: str
    search_results: str
    final_story: str

def tool_search_news(state: AgentState):
    """Busca las últimas noticias en Tavily"""
    search = TavilySearchResults(max_results=3)
    results = search.invoke(state["question"])
    return {"search_results": str(results)}

def generator_story(state: AgentState):
    """Transforma la info en una historia para jóvenes"""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    
    prompt = f"""
    Eres un narrador experto en tecnología para niños y jóvenes. 
    Tu misión es explicar las siguientes noticias de forma super simple, como si fuera una aventura o un cuento.
    
    NOTICIAS ENCONTRADAS:
    {state['search_results']}
    
    PREGUNTA DEL USUARIO:
    {state['question']}
    
    REGLAS:
    1. Usa un lenguaje muy sencillo (evita tecnicismos).
    2. Cuéntalo como una historia o metáfora.
    3. Mantén un tono emocionante y positivo.
    """
    
    response = llm.invoke(prompt)
    return {"final_story": response.content}

# Construcción del Grafo
workflow = StateGraph(AgentState)
workflow.add_node("buscador", tool_search_news)
workflow.add_node("escritor", generator_story)

workflow.set_entry_point("buscador")
workflow.add_edge("buscador", "escritor")
workflow.add_edge("escritor", END)

app_graph = workflow.compile()

# 4. Interfaz de Usuario
if not google_key or not tavily_key:
    st.warning("👈 Introduce ambas claves en la barra lateral para comenzar.")
else:
    pregunta = st.text_input("¿Qué novedad de IA quieres que te cuente hoy?", 
                             placeholder="Ej: ¿Qué es eso de Sora?")

    if pregunta:
        with st.spinner("Buscando en el mundo digital y escribiendo tu cuento..."):
            # Ejecución del Grafo
            inputs = {"question": pregunta}
            resultado = app_graph.invoke(inputs)
            
            # Mostrar resultado
            st.subheader("✨ Tu historia de IA:")
            st.write(resultado["final_story"])
            
            with st.expander("🔍 Ver fuentes de la noticia (Trazabilidad)"):
                st.write(resultado["search_results"])
