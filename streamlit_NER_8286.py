
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
text1 = """Coronaviruses are enveloped, positive single‐stranded large RNA viruses that infect humans, but also a wide range of animals. Coronaviruses were first described in 1966 by Tyrell and Bynoe, who cultivated the viruses from patients with common colds. Based on their morphology as spherical virions with a core shell and surface projections resembling a solar corona, they were termed coronaviruses (Latin: corona = crown). Four subfamilies, namely alpha‐, beta‐, gamma‐ and delta‐coronaviruses exist. While alpha‐ and beta‐coronaviruses apparently originate from mammals, in particular from bats, gamma‐ and delta‐viruses originate from pigs and birds. The genome size varies between 26 kb and 32 kb. Among the seven subtypes of coronaviruses that can infect humans, the beta‐coronaviruses may cause severe disease and fatalities, whereas alpha‐coronaviruses cause asymptomatic or mildly symptomatic infections. SARS‐CoV‐2 belongs to the B lineage of the beta‐coronaviruses and is closely related to the SARS‐CoV virus . The major four structural genes encode the nucleocapsid protein (N), the spike protein (S), a small membrane protein (SM) and the membrane glycoprotein (M) with an additional membrane glycoprotein (HE) occurring in the HCoV‐OC43 and HKU1 beta‐coronaviruses 5. SARS‐CoV‐2 is 96% identical at the whole‐genome level to a bat coronavirus."""
text2 = """The normal anatomical location of the pancreas behind the bursa omentalis, the stomach and transverse colon, starting from the curvature of the duodenum until the spleen extends transversely and upward. Lobulations considered to be variational in the pancreatic tail section were determined in a 67-year-old male patient coming to Inonu University Turgut Ozal Medical Center Urology Department with urinary burning complaint and diagnosed with benign prostatic hyperplasia. As a consequence pancreas likened to gull was named gull pancreas in the result of CT. """
text3 = """Because the boxing training program was applied specifically for patients with PD, we chose the following disease-specific outcome measures: Unified Parkinson Disease Rating Scale (UPDRS) and Parkinson Disease Quality of Life Scale (PDQL). The UPDRS was designed to monitor PD-related impairments and disabilities.27 Activities of daily living subscale II and motor examination subscale III were included in this case series. The ADL subscale measures patients' reported ability to perform everyday tasks, and the motor examination subscale measures muscular involvement of PD, indicating the severity of motor disease. Lower scores on the UPDRS signify higher levels of function. High test-retest reliability of the UPDRS (ICC=.89 for the ADL subscale; ICC=.93 for the motor examination subscale) in adults with PD has been established.22 Minimal detectable changes of 4 points on the ADL subscale and 11 points on the motor examination subscale have been established."""
text4 = """Adoption of urbanised lifestyles together with changes in reproductive behaviour might partly underlie the continued rise in worldwide incidence of breast cancer. Widespread mammographic screening and effective systemic therapies have led to a stage shift at presentation and mortality reductions in the past two decades. Loco-regional control of the disease seems to affect long-term survival, and attention to surgical margins together with improved radiotherapy techniques could further contribute to mortality gains. Developments in oncoplastic surgery and partial-breast reconstruction have improved cosmetic outcomes after breast-conservation surgery. Optimum approaches for delivering chest-wall radiotherapy in the context of immediate breast reconstruction present special challenges.Accurate methods for intraoperative assessment of sentinel lymph nodes remain a clinical priority. Clinical trials are investigating combinatorial therapies that use novel agents targeting growth factor receptors, signal transduction pathways, and tumour angiogenesis. Gene-expression profiling offers the potential to provide accurate prognostic and predictive information, with selection of best possible therapy for individuals and avoidance of overtreatment and undertreatment of patients with conventional chemotherapy. Short-term presurgical studies in the neoadjuvant setting allow monitoring of proliferative indices, and changes in gene-expression patterns can be predictive of response to therapies and long-term outcome.Clinical trials are investigating combinatorial therapies that use novel agents targeting growth factor receptors, signal transduction pathways, and tumour angiogenesis. Gene-expression profiling offers the potential to provide accurate prognostic and predictive information, with selection of best possible therapy for individuals and avoidance of overtreatment and undertreatment of patients with conventional chemotherapy. Short-term presurgical studies in the neoadjuvant setting allow monitoring of proliferative indices, and changes in gene-expression patterns can be predictive of response to therapies and long-term outcome.Clinical trials are investigating combinatorial therapies that use novel agents targeting growth factor receptors, signal transduction pathways, and tumour angiogenesis. Gene-expression profiling offers the potential to provide accurate prognostic and predictive information, with selection of best possible therapy for individuals and avoidance of overtreatment and undertreatment of patients with conventional chemotherapy. Short-term presurgical studies in the neoadjuvant setting allow monitoring of proliferative indices, and changes in gene-expression patterns can be predictive of response to therapies and long-term outcome.with selection of best possible therapy for individuals and avoidance of overtreatment and undertreatment of patients with conventional chemotherapy. Short-term presurgical studies in the neoadjuvant setting allow monitoring of proliferative indices, and changes in gene-expression patterns can be predictive of response to therapies and long-term outcome.with selection of best possible therapy for individuals and avoidance of overtreatment and undertreatment of patients with conventional chemotherapy. Short-term presurgical studies in the neoadjuvant setting allow monitoring of proliferative indices, and changes in gene-expression patterns can be predictive of response to therapies and long-term outcome."""
text5 = """Building on knowledge of normal functional anatomy, one can pose a variety of hypotheses seeking to link the observed clinical and behavioral phenomena of anxiety disorders and their neural substrates. For instance, exaggerated responsivity or sensitivity of the amygdala could mediate abnormal threat assessment (see earlier section), exaggerated fear responses including exaggerated autonomic output (see earlier section), or abnormalities in learning about danger in the environment (see earlier section). Further, in addition to vulnerabilities to anxiety conferred by intrinsic abnormality in amygdala function, abnormal amygdala responses could be secondary to insufficient vmPFC function, leading to inability to recall extinction information (see earlier section); or, secondary to abnormal hippocampal function,undermining the capacity to distinguish between safe and dangerous contexts (see earlier section). Cognitive manifestations such as worrying and obsessing are likely mediated by excessive activity in IOFC (and related regions). Interestingly, conditions characterized by globally excessive OFC activity may involve both complaints of such cognitive symptoms (mediated by IOFC) and paradoxically relatively reduced amygdala as well as autonomic responsivity (due to suppression by mOFC), perhaps characteristic of GAD (see earlier section).conditions characterized by globally excessive OFC activity may involve both complaints of such cognitive symptoms (mediated by IOFC) and paradoxically relatively reduced amygdala as well as autonomic responsivity (due to suppression by mOFC), perhaps characteristic of GAD (see earlier section).conditions characterized by globally excessive OFC activity may involve both complaints of such cognitive symptoms (mediated by IOFC) and paradoxically relatively reduced amygdala as well as autonomic responsivity (due to suppression by mOFC), perhaps characteristic of GAD (see earlier section)."""

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

