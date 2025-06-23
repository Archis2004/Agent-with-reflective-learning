import os
import io
import PyPDF2
import docx
from PIL import Image 
import google.generativeai as genai
import google.ai.generativelanguage as glm 


def prepare_file_data(uploaded_file):
    if uploaded_file is None:
        return {'type': 'error', 'content': "No file provided.", 'name': "N/A", 'error': True}

    file_name = uploaded_file.name
    
    try:
        file_bytes = uploaded_file.read()
        uploaded_file.seek(0) 
        ext = file_name.lower().split('.')[-1]

        if ext in ('png', 'jpg', 'jpeg'):
            mime_type = uploaded_file.type 
            if not mime_type or not mime_type.startswith("image/"):
                if ext == "jpg" or ext == "jpeg": mime_type = "image/jpeg"
                elif ext == "png": mime_type = "image/png"
                else: return {'type': 'error', 'content': f"Could not determine MIME type for image '{file_name}'.", 'name': file_name, 'error': True}
            
            try:
                img = Image.open(io.BytesIO(file_bytes))
                img.verify() 
                img = Image.open(io.BytesIO(file_bytes))
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                output_io = io.BytesIO()
                img_format = 'JPEG' if mime_type == 'image/jpeg' else 'PNG'
                img.save(output_io, format=img_format)
                file_bytes = output_io.getvalue()

            except Exception as pillow_e:
                return {'type': 'error', 'content': f"Invalid or corrupt image file '{file_name}': {pillow_e}", 'name': file_name, 'error': True}

            return {'type': 'image', 'content': file_bytes, 'mime_type': mime_type, 'name': file_name, 'error': None}

        elif ext == 'pdf':
            reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
            if not reader.pages:
                return {'type': 'text', 'content': f"Warning: PDF file '{file_name}' has no pages or could not be read.", 'name': file_name, 'error': 'warning'}
            text_parts = [page.extract_text() or '' for page in reader.pages]
            extracted_text = '\n'.join(text_parts).strip()
            if not extracted_text:
                return {'type': 'text', 'content': f"Warning: No text could be extracted from PDF '{file_name}'. It might be an image-based PDF or have only non-extractable content.", 'name': file_name, 'error': 'warning'}
            return {'type': 'text', 'content': extracted_text, 'name': file_name, 'error': None}

        elif ext == 'docx':
            document = docx.Document(io.BytesIO(file_bytes))
            extracted_text = '\n'.join(para.text for para in document.paragraphs).strip()
            if not extracted_text:
                 return {'type': 'text', 'content': f"Warning: No text could be extracted from DOCX '{file_name}'. It might be empty.", 'name': file_name, 'error': 'warning'}
            return {'type': 'text', 'content': extracted_text, 'name': file_name, 'error': None}
        
        else:
            return {'type': 'error', 'content': f"Unsupported file type '{ext}'. Please upload PDF, DOCX, PNG, JPG, or JPEG.", 'name': file_name, 'error': True}

    except PyPDF2.errors.PdfReadError:
        return {'type': 'error', 'content': f"Could not read PDF file '{file_name}'. It might be corrupted or password-protected.", 'name': file_name, 'error': True}
    except Exception as e:
        print(f"Unexpected error preparing file '{file_name}': {e}")
        return {'type': 'error', 'content': f"An unexpected error occurred while processing file '{file_name}': {e}", 'name': file_name, 'error': True}


def chat_with_gemini(messages_history: list, current_file_data: dict = None, model_name_to_use: str = None) -> str:

    if not model_name_to_use:
        return "Error: No Gemini model specified for chat operation."
    
    try:
        model = genai.GenerativeModel(model_name_to_use)
    except Exception as e:
        return f"Error initializing Gemini model ('{model_name_to_use}'): {e}. Check API key and model name."

    gemini_conversation_history = []
    for msg in messages_history[:-1]: 
        role = "user" if msg['author'] == 'user' else "model"
        if msg['author'] == 'system_message_ui_only': continue
        gemini_conversation_history.append(glm.Content(role=role, parts=[glm.Part(text=msg['content'])]))

    current_user_prompt_parts = []
    last_user_message_text = messages_history[-1]['content'] 

    if current_file_data and not current_file_data.get('error'):
        
        if current_file_data['type'] == 'image':
            try:
                image_part = glm.Part(inline_data=glm.Blob(
                    mime_type=current_file_data['mime_type'],
                    data=current_file_data['content']
                ))
                current_user_prompt_parts.append(image_part)

            except Exception as e:
                return f"Error preparing image data for Gemini: {e}"
        
        elif current_file_data['type'] == 'text':
            text_context = (f"Consider the following text from the uploaded file '{current_file_data['name']}':\n"
                            f"---START OF FILE TEXT---\n{current_file_data['content']}\n---END OF FILE TEXT---\n\n"
                            f"User query: ")
            last_user_message_text = text_context + last_user_message_text

    current_user_prompt_parts.append(glm.Part(text=last_user_message_text))
    
    full_conversation_for_generate = gemini_conversation_history + [glm.Content(role="user", parts=current_user_prompt_parts)]
    
    try:
        response = model.generate_content(
            full_conversation_for_generate,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7 
            )
        )
        
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            return f"Response blocked due to prompt content. Reason: {response.prompt_feedback.block_reason.name}"
        
        if not response.candidates:
             return "No response candidates received from Gemini. This might indicate an issue or blocking."

        candidate = response.candidates[0]
        
        finish_reason_name = candidate.finish_reason.name
        if finish_reason_name != "STOP" and finish_reason_name != "MAX_TOKENS":
            safety_ratings_info = ""
            if candidate.safety_ratings:
                concerning_ratings = [
                    f"{sr.category.name}: {sr.probability.name}" 
                    for sr in candidate.safety_ratings 
                    if sr.probability.name not in ['NEGLIGIBLE', 'LOW', 'PROBABILITY_UNSPECIFIED'] 
                ]
                if concerning_ratings:
                    safety_ratings_info = " Concerning safety ratings: " + ", ".join(concerning_ratings)
            return f"Response generation stopped. Reason: {finish_reason_name}.{safety_ratings_info}"
        
        if candidate.content and candidate.content.parts and candidate.content.parts[0].text:
            return candidate.content.parts[0].text
        
        else:
            if candidate.content and candidate.content.parts:
                part_types = [type(p) for p in candidate.content.parts]
                return (f"Received a non-textual or empty response part from Gemini. "
                        f"Finish reason: {finish_reason_name}. Part types: {part_types}")
            else:
                return (f"Received an empty content object from Gemini. "
                        f"Finish reason: {finish_reason_name}.")

    except Exception as e:
        print(f"Error calling Gemini API ('{model_name_to_use}'): {type(e).__name__} - {e}")
        error_detail = str(e)
        if hasattr(e, 'message'): error_detail = e.message
        elif hasattr(e, 'details'): error_detail = e.details() if callable(e.details) else e.details
        return f"Sorry, I encountered an error communicating with Gemini: {error_detail}"