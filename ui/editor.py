import streamlit as st
from streamlit_ace import st_ace
from ui.syntax_check import validate_dsl_syntax
import uuid
from ui.process import execute_dsl
import time

st.set_page_config(page_title="DSL Processing Page", page_icon="🔧")

# Custom CSS to inject
custom_css = """
<style>
    .ace-one-dark-dsl .ace_gutter {
  background: #282c34;
  color: #5c6370;
}

.ace-one-dark-dsl .ace_print-margin {
  width: 1px;
  background: #424451;
}

.ace-one-dark-dsl {
  background-color: #282c34;
  color: #abb2bf;
}

.ace-one-dark-dsl .ace_cursor {
  color: #528bff;
}

.ace-one-dark-dsl .ace_marker-layer .ace_selection {
  background: #3e4451;
}

.ace-one-dark-dsl.ace_multiselect .ace_selection.ace_start {
  box-shadow: 0 0 3px 0px #282c34;
}

.ace-one-dark-dsl .ace_marker-layer .ace_step {
  background: rgb(198, 219, 174);
}

.ace-one-dark-dsl .ace_marker-layer .ace_bracket {
  margin: -1px 0 0 -1px;
  border: 1px solid #747369;
}

.ace-one-dark-dsl .ace_marker-layer .ace_active-line {
  background: #2c313a;
}

.ace-one-dark-dsl .ace_gutter-active-line {
  background-color: #2c313a;
}

.ace-one-dark-dsl .ace_marker-layer .ace_selected-word {
  border: 1px solid #3e4451;
}

.ace-one-dark-dsl .ace_invisible {
  color: #5c6370;
}

.ace-one-dark-dsl .ace_keyword {
  color: #c678dd;
}

.ace-one-dark-dsl .ace_support.ace_function {
  color: #56b6c2;
}

.ace-one-dark-dsl .ace_support.ace_constant {
  color: #d19a66;
}

.ace-one-dark-dsl .ace_support.ace_type {
  color: #e5c07b;
}

.ace-one-dark-dsl .ace_constant.ace_numeric {
  color: #d19a66;
}

.ace-one-dark-dsl .ace_string {
  color: #98c379;
}

.ace-one-dark-dsl .ace_comment {
  color: #5c6370;
  font-style: italic;
}

.ace-one-dark-dsl .ace_punctuation.ace_operator {
  color: #abb2bf;
}

.ace-one-dark-dsl .ace_variable {
  color: #e06c75;
}
</style>
"""

# Inject custom CSS and JavaScript
st.markdown(custom_css, unsafe_allow_html=True)
if "processing_complete" not in st.session_state:
        st.session_state.processing_complete = False
        
def dsl_editor():
    st.title("Custom DSL Editor")

    # Create Ace editor instance
    content = st_ace(
        placeholder="Start typing...",
        language="custom",  # Use our custom mode
        theme="monokai",
        keybinding="vscode",
        font_size=14,
        tab_size=4,
        show_gutter=True,
        show_print_margin=False,
        wrap=True,
        auto_update=True,
        readonly=False,
        min_lines=30,
        key="dsl_editor",
        height=400,
    )

    if st.button("Process DSL"):
        st.write("Validate DSL...")
        if content.strip(): 
            is_valid, message = validate_dsl_syntax(content)
            if is_valid:
                st.success("DSL is valid. Start Training...")
                result = execute_dsl(content)
                if result:
                    st.success("DSL is Executing...")
                    time.sleep(5)
                    st.success("DSL Executed Successfully.")
                    st.session_state.processing_complete = True
            else:
                st.error(message)
                st.error("Cannot process invalid DSL. Please correct the errors and try again.")
        else:
            st.warning("The editor is empty. Please enter some DSL code before validating.")
    if st.session_state.processing_complete:
      if st.button("Show Results"):
        show_results() 

def show_results() -> None:
    with st.expander("Results", expanded=True):
        st.write("This ML training and feature importance.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Display the first image
            st.image("figures/SVM_grid_search.png", 
                     caption="SVM Grid Search",
                     use_column_width=True)
        
        with col2:
            # Display the second image
            st.image("figures/SVM_importance.png", 
                     caption="SVM Feature Importance",
                     use_column_width=True)
        
# def process_dsl(content) -> bool:
#     if "AutoML" in content:
#         return dvc_auto(content)
#     else:
#         return execute_dsl(content)
    

if __name__ == "__main__":
    dsl_editor()