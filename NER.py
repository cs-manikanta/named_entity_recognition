import streamlit as st
import nltk
import os
import spacy
from spacy import displacy
nltk.download('words')
nltk.download('punkt')
nltk.download('maxent_ne_chunker')
nltk.download('averaged_perceptron_tagger')
st.header("Named Entity Recognition")
st.markdown("Identify and Categorize Key Information from Text.")
NER_s = spacy.load("en_core_web_sm")
st.markdown("<p style='text-align: right;display: block;color: black;'>Avg. Time Taken: 0.1s</p>", unsafe_allow_html=True)
def ner_fun(text):
    t1 = ''
    t2 = []
    for sent in nltk.sent_tokenize(text):
        for chunk in nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(sent))):
            if hasattr(chunk,'label'):
                t1 =t1+"**"+' '.join(c[0] for c in chunk)+"**"+' '+'*'+chunk.label()+"*"+" "
                v = chunk.label()
                ne = ' '.join(c[0] for c in chunk)
                if not (ne in t2):
                    t2.append(ne)
                # st.write(' '.join(c[0] for c in chunk),chunk.label())
                    if v=="GPE":
                        st.markdown(f"<span style='background-color:#F7974C;font-weight:bold;padding: 5px 5px 5px 5px;color: black;border-bottom-left-radius:5%;border-bottom-right-radius:5%;border-top-right-radius:5%;border-top-left-radius:5%;font-size:120%'>{ne}</span>  : {v}", unsafe_allow_html=True)
                    elif v=="FACILITY":
                        st.markdown(f"<span style='background-color:#9179CD;font-weight:bold;padding: 5px 5px 5px 5px;color: black;border-bottom-left-radius:5%;border-bottom-right-radius:5%;border-top-right-radius:5%;border-top-left-radius:5%;font-size:120%'>{ne}</span>  : {v}", unsafe_allow_html=True)
                    elif v=="PERSON":
                        st.markdown(f"<span style='background-color:#24EAE7;font-weight:bold;padding: 5px 5px 5px 5px;color: black;border-bottom-left-radius:5%;border-bottom-right-radius:5%;border-top-right-radius:5%;border-top-left-radius:5%;font-size:120%'>{ne}</span>  : {v}", unsafe_allow_html=True)
                    elif v=="ORGANIZATION":
                        st.markdown(f"<span style='background-color:#E765F6;font-weight:bold;padding: 5px 5px 5px 5px;color: black;border-bottom-left-radius:5%;border-bottom-right-radius:5%;border-top-right-radius:5%;border-top-left-radius:5%;font-size:120%'>{ne}</span>  : {v}", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<span style='background-color:#5D4953;font-weight:bold;padding: 5px 5px 5px 5px;color: black;border-bottom-left-radius:5%;border-bottom-right-radius:5%;border-top-right-radius:5%;border-top-left-radius:5%;font-size:120%'>{ne}</span>  : {v}", unsafe_allow_html=True)
            
            else:
                t1 = t1+chunk[0]+" "
    return t1
tab1, tab2 = st.tabs(["NER from Text Input", "NER from Text File"])
# with st.expander('Get Named Entities'):
with tab1:
    text = st.text_area('Text Input: ')
    #     n_button=st.button('NLTK')
    model = st.selectbox('Select NER Model:',('Spacy', 'NLTK'))
    if text:
        if model=="NLTK":
            st.markdown(ner_fun(text))
        elif model=="Spacy":
            text1 = NER_s(text)
            ent_html = displacy.render(text1, style="ent", jupyter=False)
            # Display the entity visualization in the browser:
            st.markdown(ent_html, unsafe_allow_html=True)
            # for word in text1.ents:
            #     st.write(word.text,word.label_)

# st.markdown("##")
with tab2:
    file = st.file_uploader("File upload", type=["txt"])
    model = st.selectbox('Select NER Model:',('spacy', 'nltk'))
    if file:
        # st.write("Filename: ",file.name)
        with open(os.path.join(os.getcwd(),file.name),"wb") as f:
            f.write(file.getbuffer())
            path = os.path.join(os.getcwd(),file.name)
            f.close()
        with open(os.path.join(os.getcwd(),file.name),"r") as f:
            text = f.read()
            if text:
                if model=="nltk":
                    st.markdown(ner_fun(text))
                else:
                    text1 = NER_s(text)
                    ent_html = displacy.render(text1, style="ent", jupyter=False)
                    # Display the entity visualization in the browser:
                    st.markdown(ent_html, unsafe_allow_html=True)
                    # for word in text1.ents:
                    #     st.write(word.text,word.label_)
        os.remove(os.path.join(os.getcwd(),file.name))





    