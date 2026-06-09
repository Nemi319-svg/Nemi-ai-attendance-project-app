import streamlit as st
import numpy as np
import time
from PIL import Image

# Import your custom modules
from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from src.pipelines.face_pipeline import predict_attendance, get_face_embeddings, train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding
from src.database.db import get_all_students, create_student, get_student_subjects, get_student_attendance, unenroll_student_to_subject
from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card

def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data['student_id']
    
    c1, c2 = st.columns(2, vertical_alignment='center', gap='xxlarge')
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"Welcome, {student_data['name']}")
        if st.button("Logout", type='secondary', key='logout_btn', shortcut="control+backspace"):
            st.session_state['is_logged_in'] = False
            del st.session_state.student_data 
            st.rerun()

    st.write("") # Using write instead of space() for spacing

    c1, c2 = st.columns(2)
    with c1:
        st.header('Your Enrolled Subjects')
    with c2:
        # Changed width to use_container_width
        if st.button('Enroll in Subject', type='primary', use_container_width=True):
            enroll_dialog()

    st.divider()

    with st.spinner('Loading your enrolled subjects...'):
        subjects = get_student_subjects(student_id)
        logs = get_student_attendance(student_id)

    stats_map = {}
    for log in logs:
        sid = log['subject_id']
        if sid not in stats_map:
            stats_map[sid] = {"total": 0, "attended": 0}
        stats_map[sid]['total'] += 1
        if log.get('is_present'):
            stats_map[sid]['attended'] += 1

    cols = st.columns(2)
    for i, sub_node in enumerate(subjects):
        sub = sub_node['subjects']
        sid = sub['subject_id']
        stats = stats_map.get(sid, {"total": 0, "attended": 0})

        # Define the button with a dynamic, unique key based on the subject ID
        def create_unenroll_callback(s_id, s_name):
            def callback():
                unenroll_student_to_subject(student_id, s_id)
                st.toast(f'Unenrolled from {s_name} successfully!')
                st.rerun()
            return callback

        with cols[i % 2]:
            # Assuming subject_card handles the footer_callback correctly
            subject_card(
                name=sub['name'],
                code=sub['subject_code'],
                section=sub['section'],
                stats=[
                    ('📅', 'Total', stats['total']),
                    ('✅', 'Attended', stats['attended']),
                ],
                footer_callback=lambda: st.button(
                    "Unenroll from this course", 
                    key=f"unenroll_{sid}", 
                    type='tertiary', 
                    use_container_width=True, 
                    icon=':material/delete_forever:',
                    on_click=create_unenroll_callback(sid, sub['name'])
                )
            )
    footer_dashboard()

def student_screen():
    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return
    
    # Login Screen logic remains similar...
    # (Ensure your existing login logic is placed here)
    footer_dashboard()
