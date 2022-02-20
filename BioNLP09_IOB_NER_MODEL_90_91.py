
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
text1 = """Monocytes and macrophages play essential roles in all stages of atherosclerosis – from early precursor lesions to advanced stages of the disease. Intima-resident macrophages are among the first cells to be confronted with the influx and retention of apolipoprotein B-containing lipoproteins at the onset of hypercholesterolemia and atherosclerosis development. In this review, we outline the trafficking of monocytes and macrophages in and out of the healthy aorta, as well as the adaptation of their migratory behaviour during hypercholesterolemia. Furthermore, we discuss the functional and ontogenetic composition of the aortic pool of mononuclear phagocytes and its link to the atherosclerotic disease process. The development of mouse models of atherosclerosis regression in recent years,has enabled scientists to investigate the behaviour of monocytes and macrophages during the resolution of atherosclerosis. Herein, we describe the dynamics of these mononuclear phagocytes upon cessation of hypercholesterolemia and how they contribute to the restoration of tissue homeostasis. The aim of this review is to provide an insight into the trafficking, fate and disease-relevant dynamics of monocytes and macrophages during atherosclerosis, and to highlight remaining questions. We focus on the results of rodent studies, as analysis of cellular fates requires experimental manipulations that cannot be performed in humans but point out findings that could be replicated in human tissues.Understanding of the biology of macrophages in atherosclerosis provides an important basis for the development of therapeutic strategies to limit lesion formation and promote plaque regression."""
text2 = """The recruited Ly6Chigh monocytes are thought to primarily differentiate into intimal macrophages. Data from developing atherosclerotic plaques is lacking, but it is conceivable that Ly6Chigh monocytes have alternative fates within the lesion . As has been shown for sterile liver injury, Ly6Chigh monocytes can exhibit distinct monocyte-specific functions, including the uptake of trapped apoB-containing lipoproteins. In this way, monocytes participate in the vicious cycle of cellular apoptosis and necrosis following the metabolic stress of intracellular cholesterol accumulation. Some Ly6Chigh monocytes might also recirculate into the blood and lymph and present antigens, including de novo generated autoantigens to the cells of the adaptive immune system. Ly6Clow monocytes, on the other hand, show an intensified patrolling behaviour at atheroprone sites, which display increased endothelial damage during hypercholesterolemia. Despite their main task as patrolling intravascular cells, Ly6Clow monocytes can also be found in the atherosclerotic plaque, highlighting their potential to extravasate – although to a lesser extent than classical Ly6Chigh monocytes. These cells display an anti-inflammatory transcriptional signature with elevated transcripts for cholesterol efflux and vascular repair, thereby promoting the inflammation resolution . It is still debated whether the extravasation of Ly6Clow monocytes is of importance in the atherosclerotic disease process . If so, the anti-atherosclerotic phenotype of Ly6Clow monocytes presumably ameliorates the disease process and enhancing Ly6Clow monocyte extravasation might be a potential therapeutic target. """
text3 = """A hallmark of atherosclerosis regression is the reduction of the plaque macrophage content. Macrophage emigration from arteries via afferent lymphatics or reverse transendothelial migration aids the host defence by presenting antigens to the adaptive immune system. As described above, hypercholesterolemia blunts the CCR7-guided emigration via the expression of neuroimmune guidance cues, including netrin 1 and semaphoring 3E, and by increasing plasma membrane cholesterol content which affects intracellular signalling as well as other mechanisms . Not surprisingly, the reversal of hypercholesterolemia has been shown to induce CCR7 expression in plaque macrophages and with it their efflux via afferent lymphatics . Whether lesional macrophages leave the regressing plaque via reverse transendothelial migration, as well as the quantitative relevance of macrophage emigration to the overall loss of plaque macrophages has not yet been clarified. Increased macrophage emigration has been observed in several different models of atherosclerosis regression, including the aortic transplantation, the Reversa mouse and apoB-targeted antisense oligonucleotide treatment , whereas other reports have found no difference in macrophage emigration behaviour during regression . Importantly, emigration of plaque macrophages to lymph nodes might aid the development of the recently described post-resolution phase, although it is unknown whether this establishment of adaptive immunity takes place in atherosclerosis regression ."""
text4 = """Although the human intima harbours subendothelial macrophages and CD11c+ phagocytes that mirror the recently identified aortic intima-resident macrophage of the mouse, there are important differences in the intimal composition between mouse models and humans that need to be considered. The human intima is thickened and comprises abundant VSMCs and extracellular matrix at sites prone to atherosclerotic development. Additionally, VSMCs are very plastic cells and in addition to being producers of extracellular matrix components can be phagocytic and develop into foam cells. The discrimination of VSMC and macrophage foam cells is complicated by the fact that, VSMCs can express macrophage markers like CD68, whereas macrophages have also been found to express VSMC lineage markers. Lineage-tracing studies in mice have shown a varying degree of foam cells originating from VSMCs, ranging from 16% to 70% . In humans, it has been estimated through the analysis of histone marks that 18% of CD68+ plaque cells originate from the VSMC lineage . Future studies will have to determine to what extent VSMCs and macrophages contribute to the foam cell pool at different phases of the atherosclerotic process. """
text5 = """Besides the quantitatively limited replenishment of the macrophage pool in the adventitia, monocytes have important homeostatic functions in the vasculature. Non-classical Ly6ClowCCR2- monocytes, which derive from Ly6ChighCCR2+ in mice, crawl along the endothelium to survey the cellular integrity and sense dangers, as well as to remove cellular debris . Ly6Clow monocytes are, however, thought to rarely cross the endothelial barrier into the tissue. In contrast, classical Ly6Chigh monocytes are highly mobile and extravasate, mainly guided by their CCR2 expression. A population of transiently sessile monocytes has been found in the lungs and skin of mice in their steady-state. These ‘tissue monocytes’ have previously also been identified in the spleen, which acts as a reservoir to quickly mobilize immune cells upon inflammation. Contrary to splenic monocytes, the Ly6Chigh monocytes in lung and skin can survey the tissue environment and transport antigens to lymph nodes. Although the question of sessile monocyte existence has not been addressed for the arterial wall, monocytes are readily identified in the healthy arterial wall. Their homeostatic turnover and ability to recirculate into the blood or leave via afferent lymphatics into adjacent lymph nodes, similarly to the surveying monocytes in lung and skin, remains to be determined."""

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

