
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
text1 = """CD36 is a class B scavenger receptor observed in many cell types and tissues throughout the body. Recent literature has implicated CD36 in the pathogenesis of metabolic dysregulation such as found in obesity, insulin resistance, and atherosclerosis. Genetic variation at the CD36 loci have been associated with obesity and lipid components of the metabolic syndrome, with risk of heart disease and type 2 diabetes. Recently, non-cell bound CD36 was identified in human plasma and was termed soluble CD36 (sCD36). In this review we will describe the functions of CD36 in tissues and address the role of sCD36 in the context of the metabolic syndrome. We will also highlight recent findings from human genetic studies looking at the CD36 locus in relation to metabolic profile in the general population. Finally, we present a model in which insulin resistance, oxLDL, low-grade inflammation and liver steatosis may contribute to elevated levels of sCD36."""
text2 = """IMT, FLI, LF%, presence of the metabolic syndrome, impaired glucose regulation, insulin and triglycerides increased across sCD36 quartiles (Q2-Q4), whereas adiponectin and M/I decreased (P ≤ 0.01). sCD36 was lower in women than in men (P = 0.045). Log sCD36 showed a bimodal distribution, and amongst subjects with sCD36 within the log-normal distribution (log-normal population, n = 1029), sCD36 was increased in subjects with impaired glucose regulation (P = 0.045), metabolic syndrome (P = 0.006) or increased likelihood of fatty liver (P < 0.001). sCD36 correlated significantly with insulin, triglycerides, M/I and FLI (P < 0.05) after adjustment for study centre, gender, age, glucose tolerance status, smoking habits and alcohol consumption. In the log-normal population, these relationships were stronger than in the total study population and, additionally, sCD36 was significantly associated with LF% and IMT (P < 0.05). """
text3 = """The prevalence of metabolic syndrome was 34.6% (male, 31.2%; female, 37.3%) (P > 0.05) and 28.8% (male, 23.1%; female, 33.5% (P < 0.01) according to IDF criteria and ATP III, respectively. The highest prevalence of metabolic syndrome was present in subjects aged 60-69 years; in obese people (43.2%, P < 0.001); in Hatay province (36.5%, P < 0.001); and in districts (32.2%, P > 0.05). The prevalence of metabolic syndrome criteria in all 4 provinces was as follows: type 2 diabetes mellitus, 15%; hypertension, 41.4%; obesity, 44.1%; abdominal obesity, 56.8%; low HDL-C, 34.1%; hypertriglyceridemia, 35.9%; and high LDL-C, 27.4%."""
text4 = """Although the human intima harbours subendothelial macrophages and CD11c+ phagocytes that mirror the recently identified aortic intima-resident macrophage of the mouse, there are important differences in the intimal composition between mouse models and humans that need to be considered. The human intima is thickened and comprises abundant VSMCs and extracellular matrix at sites prone to atherosclerotic development. Additionally, VSMCs are very plastic cells and in addition to being producers of extracellular matrix components can be phagocytic and develop into foam cells. The discrimination of VSMC and macrophage foam cells is complicated by the fact that, VSMCs can express macrophage markers like CD68, whereas macrophages have also been found to express VSMC lineage markers. Lineage-tracing studies in mice have shown a varying degree of foam cells originating from VSMCs, ranging from 16% to 70% . In humans, it has been estimated through the analysis of histone marks that 18% of CD68+ plaque cells originate from the VSMC lineage . Future studies will have to determine to what extent VSMCs and macrophages contribute to the foam cell pool at different phases of the atherosclerotic process. """
text5 = """The authors then zoom in on the blue regions and make their most provocative observation: they identify what they believe to be VSMC-derived macrophages. There is no doubt that blue cells inhabit the same region as CD68+ and Mac2+ cells. But are these blue cells macrophages? In fact, it appears as if CD68 and Mac2 register with regions of the intima that are largely devoid of blue, and conversely, many blue cells appear to be negative for Mac-2 and CD68. Future confocal microscopy, a technique that can better decipher whether the same cell stains for several markers in tissue, will be needed to answer whether macrophages derive from the VSMC lineage. A related problem is the choice of Mac-2 and CD68. Mac-2 (also known as galectin-3/Lgals3) is highly expressed on myeloid cells of just about every flavor. It is also expressed on different T cells, including γδT cells, as well as various stromal cells. Likewise, CD68 is expressed on the entire myeloid lineage, including neutrophils. Even if the Mac-2+ and CD68+ cells are blue, are they really macrophages? The fact that blue VSMC-derived cells do not stain for iNOS or Arg-1 again argues against the stated conclusion of the paper."""

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

    nerModel = NerDLModel.load("/content/drive/MyDrive/BioNLP09_IOB_NER_Model/Ner_glove_100d_e9_b10_lr0.02_90_91")\
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

