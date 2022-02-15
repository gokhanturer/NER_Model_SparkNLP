
import streamlit as st
import pandas as pd
import base64
import os

import sparknlp
from pyspark.ml import Pipeline,PipelineModel
from pyspark.sql import SparkSession

from sparknlp.annotator import *
from sparknlp.base import *

from sparknlp_display import NerVisualizer
from sparknlp.base import LightPipeline

spark = sparknlp.start(gpu = True) 


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

st.sidebar.image('https://nlp.johnsnowlabs.com/assets/images/logo.png', use_column_width=True)
st.sidebar.header('Choose the pretrained model')
select_model = st.sidebar.selectbox("select model name",["ner_model_glove_100d"])
if select_model == "ner_model_glove_100d":
    st.write("You choosed : ner_model_glove_100d ")

st.title("Spark NLP NER Model Playground")

#data
text1 = """Despite the success of vaccination programs for polio and some childhood diseases, other diseases like HIV/AIDS, malaria, tuberculosis, acute respiratory infections and diarrheal disease are causing high mortality rates in Africa. However, mortality figures give only a partial measure of the toll asked by infectious diseases, and the global burden includes also health impact measured by disabilities, deformities, loss of productivity, care and treatment caused by a multitude of diseases like lymphatic filariasis, leishmaniasis, schistosomiasis, sleeping sickness and others. The impact of infectious diseases can be traced according to economic performance of African countries, showing that 34 out of 53 countries are classified as low-income economies."""
text2 = """The normal anatomical location of the pancreas behind the bursa omentalis, the stomach and transverse colon, starting from the curvature of the duodenum until the spleen extends transversely and upward. Lobulations considered to be variational in the pancreatic tail section were determined in a 67-year-old male patient coming to Inonu University Turgut Ozal Medical Center Urology Department with urinary burning complaint and diagnosed with benign prostatic hyperplasia. As a consequence pancreas likened to gull was named gull pancreas in the result of CT. """
text3 = """Because the boxing training program was applied specifically for patients with PD, we chose the following disease-specific outcome measures: Unified Parkinson Disease Rating Scale (UPDRS) and Parkinson Disease Quality of Life Scale (PDQL). The UPDRS was designed to monitor PD-related impairments and disabilities.27 Activities of daily living subscale II and motor examination subscale III were included in this case series. The ADL subscale measures patients' reported ability to perform everyday tasks, and the motor examination subscale measures muscular involvement of PD, indicating the severity of motor disease. Lower scores on the UPDRS signify higher levels of function. High test-retest reliability of the UPDRS (ICC=.89 for the ADL subscale; ICC=.93 for the motor examination subscale) in adults with PD has been established.22 Minimal detectable changes of 4 points on the ADL subscale and 11 points on the motor examination subscale have been established."""
text4 = """Shigellosis, anyone can get it, but it most often affects young children. It spreads through contact with contaminated water or food or with an infected person’s poop. Because of this, it can happen in daycare centers or public swimming pools. It can cause fever, stomach pain, or diarrhea that can sometimes be bloody -- or there can be no symptoms at all. It often goes away without medication, but the diarrhea can cause dehydration. """
text5 = """Chronic diseases are long lasting conditions with persistent effects. Their social and economic consequences can impact on peoples’ quality of life. Chronic conditions are becoming increasingly common and are a priority for action in the health sector.Many people with chronic conditions do not have a single, predominant condition, but rather they experience multimorbidity—the presence of 2 or more chronic conditions in a person at the same time.AIHW commonly reports on 10 major chronic condition groups: arthritis, asthma, back pain, cancer, cardiovascular disease, chronic obstructive pulmonary disease, diabetes, chronic kidney disease, mental health conditions and osteoporosis.These chronic conditions were selected for reporting because they are common, pose significant health problems, have been the focus of ongoing AIHW surveillance efforts and, in many instances, action can be taken to prevent their occurrence.More reports and statistics on chronic disease can be found under Burden of disease, Biomedical risk factors and Life expectancy & deaths."""

sample_text = st.selectbox("",[text1, text2, text3,text4,text5])

@st.cache(hash_funcs={"_thread.RLock": lambda _: None},allow_output_mutation=True, suppress_st_warning=True)
def model_pipeline():
    documentAssembler = DocumentAssembler()\
          .setInputCol("text")\
          .setOutputCol("document")

    sentenceDetector = SentenceDetector()\
          .setInputCols(['document'])\
          .setOutputCol('sentence')

    tokenizer = Tokenizer()\
          .setInputCols(['sentence'])\
          .setOutputCol('token')

    gloveEmbeddings = WordEmbeddingsModel.pretrained()\
          .setInputCols(["document", "token"])\
          .setOutputCol("embeddings")

    nerModel = NerDLModel.load("/content/drive/MyDrive/NER_Model/Ner_glove_100d_e40_b10_lr0.02")\
          .setInputCols(["sentence", "token", "embeddings"])\
          .setOutputCol("ner")

    nerConverter = NerConverter()\
          .setInputCols(["document", "token", "ner"])\
          .setOutputCol("ner_chunk")
 
    pipeline_dict = {
          "documentAssembler":documentAssembler,
          "sentenceDetector":sentenceDetector,
          "tokenizer":tokenizer,
          "gloveEmbeddings":gloveEmbeddings,
          "nerModel":nerModel,
          "nerConverter":nerConverter
    }
    return pipeline_dict

model_dict = model_pipeline()

def load_pipeline():
    nlp_pipeline = Pipeline(stages=[
                   model_dict["documentAssembler"],
                   model_dict["sentenceDetector"],
                   model_dict["tokenizer"],
                   model_dict["gloveEmbeddings"],
                   model_dict["nerModel"],
                   model_dict["nerConverter"]
                   ])

    empty_data = spark.createDataFrame([['']]).toDF("text")

    model = nlp_pipeline.fit(empty_data)

    return model


ner_model = load_pipeline()

def viz (annotated_text, chunk_col):
  raw_html = NerVisualizer().display(annotated_text, chunk_col, return_html=True)
  sti = raw_html.find('<style>')
  ste = raw_html.find('</style>')+8
  st.markdown(raw_html[sti:ste], unsafe_allow_html=True)
  st.write(HTML_WRAPPER.format(raw_html[ste:]), unsafe_allow_html=True)


def get_entities (ner_pipeline, text):
    
    light_model = LightPipeline(ner_pipeline)

    full_annotated_text = light_model.fullAnnotate(text)[0]

    st.write('')
    st.subheader('Entities')

    chunks=[]
    entities=[]
    
    for n in full_annotated_text["ner_chunk"]:

        chunks.append(n.result)
        entities.append(n.metadata['entity'])

    df = pd.DataFrame({"chunks":chunks, "entities":entities})

    viz (full_annotated_text, "ner_chunk")
    
    st.subheader("Dataframe")
    st.write('')

    st.table(df)
    
    return df


entities_df  = get_entities (ner_model, sample_text)

