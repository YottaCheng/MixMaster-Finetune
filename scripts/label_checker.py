import streamlit as st
import json
import pandas as pd
import os

# Configuration
st.set_page_config(
    page_title="Audio Parameter Annotation Tool",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items=None
)

# Custom CSS to fix layout issues and improve aesthetics
st.markdown("""
<style>
    /* Fix padding and centering - increase top padding */
    .block-container {
        padding-top: 3.5rem !important;
        padding-bottom: 1rem !important;
        max-width: 60rem !important;
        margin: 0 auto !important;
    }
    
    /* Page title styling - add more top margin/padding */
    h1 {
        margin-top: 1.5rem !important;
        padding-top: 1.5rem !important;
        padding-bottom: 1rem !important;
        text-align: center !important;
        font-weight: 600 !important;
        color: #333 !important;
        font-size: 1.8rem !important;
    }
    
    /* Make inputs more compact */
    .stTextInput, .stFileUploader, .stNumberInput {
        margin-bottom: 0.8rem !important;
    }
    
    /* Make the multiselect more attractive but compact */
    .stMultiSelect [data-baseweb="tag"] {
        background-color: #ff6b6b;
        border-radius: 4px;
        margin-right: 5px;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    /* Improve the ID/text display */
    .entry-header {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 6px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #4682b4;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Button styling - more compact */
    .stButton > button {
        font-weight: 500 !important;
        border-radius: 4px !important;
        height: 2.2rem !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1) !important;
    }
    
    /* Center button text */
    .stButton > button > div {
        display: flex !important;
        justify-content: center !important;
        align-items: center !important;
    }
    
    /* Remove extra space in containers */
    .stVerticalBlock {
        gap: 0.5rem !important;
    }
    
    /* Make checkboxes more compact */
    .stCheckbox {
        margin-bottom: 0;
    }
    
    /* File uploader styling - more compact */
    .stFileUploader {
        border-radius: 4px !important;
        border: 1px dashed #ccc !important;
        padding: 0.5rem !important;
    }
    
    /* Input fields styling */
    .stTextInput > div > div {
        border-radius: 4px !important;
    }
    
    /* Streamlit container styling */
    .main .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Prevent horizontal overflow */
    .stSlider {
        padding-right: 1rem !important;
    }
    
    /* More compact main padding */
    .main {
        padding: 1rem !important;
    }
</style>
""", unsafe_allow_html=True)

# Label options (统一为小写)
LABEL_OPTIONS = [
    "low frequency", "mid frequency", "high frequency",
    "reverb", "effects", "sound field", "compression", "volume"
]

# Label mapping (支持大小写统一)
LABEL_MAPPING = {
    "低频": "low frequency",
    "中频": "mid frequency",
    "高频": "high frequency",
    "混响": "reverb",
    "效果器": "effects",
    "声场": "sound field",
    "压缩": "compression",
    "音量": "volume"
}
REVERSE_LABEL_MAPPING = {v: k for k, v in LABEL_MAPPING.items()}  # Reverse mapping

# Display labels in more readable form for UI
DISPLAY_LABELS = {
    "low frequency": "Low Frequency",
    "mid frequency": "Mid Frequency",
    "high frequency": "High Frequency",
    "reverb": "Reverb",
    "effects": "Effects",
    "sound field": "Sound Field",
    "compression": "Compression",
    "volume": "Volume"
}

def load_data(uploaded_file, sample_ratio):
    try:
        uploaded_file.seek(0)
        data = json.load(uploaded_file)
        
        # Verify data structure
        if not isinstance(data, list):
            st.error("Invalid format: JSON should be a list of entries")
            return pd.DataFrame(), None
        
        # Check required fields
        required = ['id', 'text', 'labels']
        for entry in data[:5]:
            if not all(k in entry for k in required):
                st.error(f"Missing required fields: {required}")
                return pd.DataFrame(), None
                
        df = pd.DataFrame(data).set_index('id')
        
        # Map Chinese labels to English and convert to lowercase
        df['labels'] = df['labels'].apply(
            lambda x: [LABEL_MAPPING.get(label, label.lower()) if isinstance(label, str) else label for label in x] if isinstance(x, list) else []
        )
        
        # Normalize all labels to lowercase for consistent comparison
        df['labels'] = df['labels'].apply(
            lambda labels: [label.lower() if isinstance(label, str) else label for label in labels]
        )
        
        # Random sampling based on sample ratio
        if sample_ratio < 100:
            df = df.sample(frac=sample_ratio / 100, random_state=42)  # Fixed seed for reproducibility
        
        return df, data  # Return both DataFrame and original data
        
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        return pd.DataFrame(), None

def merge_edits(original_data, edited_data):
    """
    Merge edited data back into the original dataset.
    """
    edited_dict = {item['id']: item['labels'] for item in edited_data}
    
    for item in original_data:
        if item['id'] in edited_dict:
            # Update labels with edited values
            item['labels'] = edited_dict[item['id']]
    
    return original_data

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = 'config'
if 'edited_data' not in st.session_state:
    st.session_state.edited_data = []
if 'original_data' not in st.session_state:
    st.session_state.original_data = None
if 'df' not in st.session_state:
    st.session_state.df = None
if 'output_path' not in st.session_state:
    st.session_state.output_path = None
if 'page_num' not in st.session_state:
    st.session_state.page_num = 0

# Configuration Page
if st.session_state.page == 'config':
    # Add extra space at the top before the title
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    st.title("Audio Parameter Annotation Tool")
    
    # Create a centered container with maximum width
    container_col1, container_col2, container_col3 = st.columns([1, 10, 1])
    
    with container_col2:
        # Vertical arrangement of parameters
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Input file selection
        input_file = st.file_uploader(
            "Upload Input File",
            type=['json'],
            help="Upload the JSON file you want to check and edit."
        )
        
        # Output directory selection
        output_dir = st.text_input(
            "Output Directory",
            value=os.getcwd(),
            help="Specify the directory where the modified file will be saved."
        )
        
        # Sampling ratio with both slider and number input
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Use columns to place slider and number input side by side
        slider_col, number_col = st.columns([3, 1])
        
        with slider_col:
            sample_ratio = st.slider(
                "Sampling Ratio (%)",
                min_value=0,
                max_value=100,
                value=100,
                help="Select the percentage of data to sample. Set to 100% for full dataset."
            )
        
        with number_col:
            manual_ratio = st.number_input(
                "Custom %",
                min_value=0,
                max_value=100,
                value=sample_ratio,
                step=1,
                help="Manually enter sample percentage"
            )
            # Sync the values
            sample_ratio = manual_ratio
        
        # Add some space before the button
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create a centered button that's not too wide
        button_col1, button_col2, button_col3 = st.columns([1, 1, 1])
        with button_col2:
            proceed = st.button("Proceed to Annotation", use_container_width=True)
        
        if proceed and input_file:
            try:
                # Load and validate data
                df, original_data = load_data(input_file, sample_ratio)
                
                if not df.empty and original_data is not None:
                    # Store data in session state
                    st.session_state.df = df
                    st.session_state.original_data = original_data
                    st.session_state.output_path = os.path.join(output_dir, 'modified_data.json')
                    st.session_state.page = 'annotation'
                    st.session_state.page_num = 0
                    st.session_state.edited_data = []
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")

# Annotation Page
elif st.session_state.page == 'annotation':
    # Add extra space at the top before the title
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    
    st.title("Audio Parameter Annotation Tool")
    
    # Add return to config button in sidebar
    with st.sidebar:
        st.header("Control Panel")
        if st.button("⬅️ Return to Configuration", use_container_width=True):
            st.session_state.page = 'config'
            st.rerun()
    
    # Load data from session state
    df = st.session_state.df
    original_data = st.session_state.original_data
    output_path = st.session_state.output_path
    
    if df is None or df.empty or original_data is None:
        st.error("No data available. Please return to configuration.")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Return to Configuration", use_container_width=True):
                st.session_state.page = 'config'
                st.rerun()
        st.stop()
    
    # Create a centered container
    main_col1, main_col2, main_col3 = st.columns([1, 10, 1])
    
    with main_col2:
        # Calculate pagination
        total_pages = (len(df) - 1) // 3 + 1
        
        # Navigation buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("⬅️ Previous", use_container_width=True) and st.session_state.page_num > 0:
                st.session_state.page_num -= 1
        with col2:
            st.markdown(f"<h3 style='text-align: center;'>Page {st.session_state.page_num + 1} of {total_pages}</h3>", unsafe_allow_html=True)
        with col3:
            if st.button("Next ➡️", use_container_width=True) and st.session_state.page_num < total_pages - 1:
                st.session_state.page_num += 1
    
    # Display current page data
    start_idx = st.session_state.page_num * 3
    end_idx = min(start_idx + 3, len(df))
    current_data = df.iloc[start_idx:end_idx]
    
    # Process each item in the current page
    for i, (row_id, row) in enumerate(current_data.iterrows()):
        # Create a container for each entry with less vertical space
        with st.container():
            # Display ID and text with better formatting
            st.markdown(f"""
            <div class="entry-header">
                <strong>ID: {row_id}</strong><br>
                {row['text']}
            </div>
            """, unsafe_allow_html=True)
            
            # Position the skip checkbox on the right
            col1, col2 = st.columns([3, 1])
            with col2:
                skip_edit = st.checkbox(f"Skip this entry", key=f"skip_{row_id}")
            
            if not skip_edit:
                # Normalize labels for validation (convert to lowercase)
                normalized_labels = [label.lower() if isinstance(label, str) else label for label in row['labels']]
                
                # Check for valid labels - case insensitive comparison
                valid_labels = [label for label in normalized_labels if label in LABEL_OPTIONS]
                
                # Show warning if invalid labels are found
                if len(valid_labels) != len(normalized_labels):
                    invalid_labels = set(normalized_labels) - set(LABEL_OPTIONS)
                    if invalid_labels:
                        st.warning(f"⚠️ Invalid labels found for ID {row_id}: {', '.join(invalid_labels)}")
                
                # Set up multiselect with proper display labels
                with col1:
                    # Map internal labels to display labels for UI
                    display_defaults = [DISPLAY_LABELS.get(label, label.title()) for label in valid_labels]
                    display_options = list(DISPLAY_LABELS.values())
                    
                    # Create the multiselect with display labels
                    selected_display_labels = st.multiselect(
                        "Select Labels",
                        options=display_options,
                        default=display_defaults,
                        key=f"label_{row_id}"
                    )
                    
                    # Map display labels back to internal format
                    reverse_display_map = {v: k for k, v in DISPLAY_LABELS.items()}
                    selected_internal_labels = [reverse_display_map.get(label, label.lower()) for label in selected_display_labels]
                    
                    # Map to Chinese for saving
                    mapped_labels = [REVERSE_LABEL_MAPPING.get(label, label) for label in selected_internal_labels]
                    
                    # Store the edited data
                    # Check if we already have an entry for this ID
                    existing_entry = next((item for item in st.session_state.edited_data if item["id"] == row_id), None)
                    
                    if existing_entry:
                        # Update existing entry
                        existing_entry["labels"] = mapped_labels
                    else:
                        # Add new entry
                        st.session_state.edited_data.append({
                            "id": row_id,
                            "labels": mapped_labels
                        })
            
            # Small space between entries
            st.markdown("<div style='margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    # Add space before save button
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Save button - centered
    save_col1, save_col2, save_col3 = st.columns([1, 2, 1])
    with save_col2:
        if st.button("💾 Save Changes", use_container_width=True):
            try:
                # Merge edits back into original data
                merged_data = merge_edits(original_data, st.session_state.edited_data)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Save the merged data
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(merged_data, f, indent=2, ensure_ascii=False)
                
                st.success(f"✅ File saved successfully at {output_path}")
            
            except Exception as e:
                st.error(f"❌ Save failed: {str(e)}")
    
    # Display progress in sidebar
    with st.sidebar:
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Progress")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Entries", len(df))
        with col2:
            st.metric("Edited", len(st.session_state.edited_data))
        
        st.progress((st.session_state.page_num + 1)/total_pages)
        
        # Quick navigation
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("⏩ Quick Navigate")
        
        # Add quick navigation if many pages
        if total_pages > 5:
            jump_to = st.number_input("Jump to page", 1, total_pages, st.session_state.page_num + 1)
            if st.button("Go", use_container_width=True):
                st.session_state.page_num = jump_to - 1
                st.rerun()