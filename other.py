import streamlit as st

def about_stroke():
    st.header("How dangerous is stroke?")
    st.subheader("Did you know?")

    st.write("""
    
    Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Commodo viverra maecenas accumsan lacus vel facilisis volutpat est velit. Tristique et egestas quis ipsum suspendisse ultrices gravida. Sit amet aliquam id diam maecenas ultricies. Sed risus ultricies tristique nulla aliquet enim tortor at auctor. Sit amet porttitor eget dolor morbi non arcu risus quis. Ac tortor dignissim convallis aenean et. Nec feugiat in fermentum posuere urna nec tincidunt praesent semper. Eget lorem dolor sed viverra ipsum. Convallis aenean et tortor at risus viverra adipiscing at. Tellus rutrum tellus pellentesque eu tincidunt tortor aliquam nulla. Risus ultricies tristique nulla aliquet enim tortor at auctor. Vel elit scelerisque mauris pellentesque pulvinar pellentesque habitant morbi tristique. Amet porttitor eget dolor morbi non arcu. Elementum pulvinar etiam non quam lacus suspendisse. Enim nunc faucibus a pellentesque sit amet. Urna id volutpat lacus laoreet non curabitur gravida. Sed arcu non odio euismod lacinia at quis risus. Tincidunt eget nullam non nisi est sit amet. Duis ultricies lacus sed turpis tincidunt id aliquet risus feugiat.

    Quam adipiscing vitae proin sagittis nisl rhoncus mattis rhoncus urna. Suspendisse in est ante in. Purus gravida quis blandit turpis cursus in hac habitasse platea. A lacus vestibulum sed arcu non. Lacinia quis vel eros donec ac odio. Bibendum ut tristique et egestas quis ipsum. Turpis egestas integer eget aliquet nibh. Tincidunt vitae semper quis lectus nulla at volutpat. Malesuada fames ac turpis egestas. Dolor morbi non arcu risus quis varius quam quisque. Massa tincidunt dui ut ornare lectus sit. In fermentum et sollicitudin ac orci phasellus egestas. Dictum sit amet justo donec enim diam. Aliquam eleifend mi in nulla posuere sollicitudin aliquam ultrices. Elit at imperdiet dui accumsan sit amet nulla facilisi morbi. Dictum non consectetur a erat nam at lectus. Nec dui nunc mattis enim ut. Vel elit scelerisque mauris pellentesque pulvinar pellentesque habitant morbi tristique. Scelerisque fermentum dui faucibus in.

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