import streamlit as st
import stanza
import pandas as pd

# إعداد الصفحة
st.set_page_config(page_title="محلل الجمل العربية", page_icon="📝")

# تحميل المحرك العربي (سأجعله يحمل مرة واحدة فقط)
@st.cache_resource
def load_arabic_model():
    stanza.download('ar')
    return stanza.Pipeline('ar')

try:
    nlp = load_arabic_model()
except:
    st.error("حدث خطأ في تحميل المحرك، يرجى إعادة المحاولة.")

# قاموس لتحويل المصطلحات التقنية للعربية
labels = {
    "NOUN": "اسم", "VERB": "فعل", "ADJ": "صفة", "PRON": "ضمير",
    "ADP": "حرف جر", "CCONJ": "حرف عطف", "ADV": "ظرف", "DET": "أداة تعريف",
    "nsubj": "فاعل", "obj": "مفعول به", "root": "الركن الأساسي",
    "obl": "شبه جملة", "amod": "نعت", "nmod": "مضاف إليه"
}

st.title("🎯 تطبيق تحليل الجمل العربية")
st.write("هذا التطبيق يقوم بإعراب الجملة وتفكيكها لمكوناتها الأساسية.")

sentence = st.text_input("اكتب جملتك هنا:", "ذهب الطالبُ إلى المدرسةِ")

if st.button("حلل الجملة الآن"):
    if sentence:
        with st.spinner('انتظر قليلاً، جاري الإعراب...'):
            doc = nlp(sentence)
            results = []
            for sent in doc.sentences:
                for word in sent.words:
                    results.append({
                        "الكلمة": word.text,
                        "الجذر": word.lemma,
                        "النوع": labels.get(word.upos, word.upos),
                        "الوظيفة النحوية": labels.get(word.deprel, word.deprel)
                    })
            st.success("تم التحليل!")
            st.table(pd.DataFrame(results))
    else:
        st.warning("يرجى كتابة جملة أولاً.")