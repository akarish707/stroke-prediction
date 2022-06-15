import streamlit as st

def about_stroke():
    
    col11, col12 = st.columns([2,2])
    
    with col11:
        st.write("# What is stroke?")
        # st.image("asset/image/stock-photo-walk-training-and-rehabilitation-process-old-asian-stroke-patient-learning-to-use-walker-with-1781080226.jpg")
    
    with col12:
        st.write("""
        
        Knowing the signs of a stroke is the first step in stroke prevention. **A stroke, sometimes called a "brain attack,"** occurs when blood flow to an area in the brain is cut off. The brain cells, deprived of the oxygen and glucose needed to survive, die. 
        ##### If a stroke is not caught early, permanent brain damage or death can result.

        """)

    col21,col22 = st.columns([1,2]) 
    with col21:
        st.write("## What are the Symptoms of Stroke?")
    
    with col22:
        st.write("""
            **Weakness or numbness** of the face, arm, or leg on one side of the body. **Loss of vision** or dimming (like a curtain falling) in one or both eyes.
            **Loss of speech**, difficulty talking, or understanding what others are saying.
            Sudden, **severe headache** with no known cause.
            **Loss of balance** or **unstable walking**, usually combined with another symptom.

            """)
    
    col31, col32= st.columns([2,3])
    with col31:
        st.write("## Controllable Risk Factors for Stroke:")

    
    with col32:
        st.write("""
        - [High blood pressure](https://www.webmd.com/hypertension-high-blood-pressure/ss/slideshow-hypertension-overview)
        - [Atrial fibrillation](https://www.webmd.com/heart-disease/atrial-fibrillation/a-fib-overview)
        - [Uncontrolled diabetes](https://www.webmd.com/cholesterol-management/cholesterol-assessment/default.htm)
        - High cholesterol
        - Smoking
        - Excessive alcohol intake
        - Obesity
        - Carotid or [coronary artery disease](https://www.webmd.com/heart-disease/heart-disease-coronary-artery-disease)
        """)

    col41, col42 = st.columns([1,2])
    with col41:
        st.write("## Uncontrollable Risk Factors for Stroke:")
    
    with col42:
        st.write("""
    > ##### Age (>65)

    > ##### Gender (Men have more strokes, but women have deadlier strokes)

    > ##### Race (African-Americans are at increased risk)
    
    > ##### Family history of stroke
    
    """)

    st.write("""
    ## More info...
    
    > Your doctor can evaluate your risk for stroke and help you control your risk factors. Sometimes, people experience warning signs before a stroke occurs.

    >These are called transient ischemic attacks (also called TIA or "mini-stroke") and are short, brief episodes of the stroke symptoms listed above. Some people have no symptoms warning them prior to a stroke or symptoms are so mild they are not noticeable. Regular check-ups are important in catching problems before they become serious. Report any symptoms or risk factors to your doctor.
    
    ##### You can check your possibility of your stroke by navigating to our "Predict your health menu" [above](#stroke-prediction)!

    """)
    
    st.write("""
    ---
    **Source:**

    ##### [Heart Disease and Stroke (WebMD)](https://www.webmd.com/heart-disease/stroke)
        """)


def about_us():
    st.header("We are a team of students👩‍🎓👨‍🎓")
    left,right = st.columns([1,8])

    with left:
        st.image("asset/image/crying_kid.gif")

    with right:
        st.subheader("*sleepy students to be exact*")
    

    st.write("---")

    colpic1, coldesc1 = st.columns([1,3])
    with colpic1:
        st.image("asset/image/cheryl.jpeg")

    with coldesc1:
        st.subheader("Cheryl Almeira")
        st.write("""
        A passionate student with relentless idea to make everything around her better in everyway.
        """)
    st.write("---")
    
    colpic2, coldesc2 = st.columns([3,1])
    with colpic2:
        st.subheader("Michelle A. Guntoro")
        
        st.write("""
        
        Giving up on a great idea is not on her dictionary. Her team management skills is the main driver of the project. 
        """)

    with coldesc2:
        st.image("asset/image/michelle.jpeg")
        

    st.write("---")
    
    colpic3, coldesc3= st.columns([1,3])
    with colpic3:
        st.image("asset/image/vito.jpeg")
        
    with coldesc3:

        st.subheader("Vito P. Minardi")

        st.write("""
        
        Idea runs in his blood. Should he be born a hundred years ago, he could've been the picasso of the century.
        """)
    st.write("---")