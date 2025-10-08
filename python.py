# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(col, errors='coerce', downcast='integer').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Nếu không tìm thấy, cố gắng tìm Tổng Tài sản để tránh lỗi nghiêm trọng
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* XỬ LÝ CHIA CHO 0 *******************************
    # Đảm bảo mẫu số không phải 0 để tính Tỷ trọng
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* KẾT THÚC XỬ LÝ *******************************
    
    return df

# --- Hàm gọi API Gemini cho Nhận xét Tự động ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm khởi tạo Chat Session (Dùng Cache để duy trì phiên) ---
@st.cache_resource
def setup_gemini_chat(data_context):
    """Khởi tạo client và đối tượng chat với ngữ cảnh dữ liệu."""
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            # Không dùng st.error ở đây để không làm gián đoạn flow
            return None
            
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 
        
        # Cung cấp ngữ cảnh là dữ liệu tài chính đã được xử lý
        system_instruction = f"""
        Bạn là một trợ lý phân tích tài chính chuyên nghiệp. Nhiệm vụ của bạn là trả lời các câu hỏi 
        của người dùng dựa trên Báo cáo Tài chính đã được xử lý và tính toán sau:
        
        DỮ LIỆU TÀI CHÍNH ĐÃ PHÂN TÍCH:
        {data_context}
        
        Hãy giữ câu trả lời ngắn gọn, chính xác, và chỉ dựa vào dữ liệu trên. 
        Nếu người dùng hỏi về dữ liệu không có, hãy trả lời là bạn không biết.
        """
        
        # Bắt đầu phiên trò chuyện với hướng dẫn hệ thống
        chat = client.chats.create(
            model=model_name,
            system_instruction=system_instruction
        )
        return chat
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Chat: {e}")
        return None


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn và Nợ ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Xử lý lỗi chia cho 0
                if no_ngan_han_N != 0:
                    thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                
                if no_ngan_han_N_1 != 0:
                    thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                # Hiển thị Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else thanh_toan_hien_hanh_N_1
                    )
                with col2:
                    delta_value = "N/A"
                    if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float):
                         delta_value = f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                         
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần" if isinstance(thanh_toan_hien_hanh_N, float) else thanh_toan_hien_hanh_N,
                        delta=delta_value if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float) else None
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số. Chỉ số được đặt là 'N/A'.")
            except ZeroDivisionError:
                 st.warning("Giá trị Nợ Ngắn Hạn bằng 0. Chỉ số Thanh toán Hiện hành không tính được (Infinite/N/A).")
                 thanh_toan_hien_hanh_N = "N/A"
                 thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI (Dùng cho cả Chức năng 5 và 6)
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    # Dùng try/except an toàn hơn cho .iloc[0]
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)].empty else "N/A", 
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A", 
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")


            # ====================================================================================
            # --- CHỨC NĂNG MỚI: 6. KHUNG CHAT HỎI ĐÁP AI (GIỮ NGUYÊN CODE BỔ SUNG CỦA BẠN) ---
            # ====================================================================================
            st.divider() # Phân tách với phần cũ
            st.subheader("6. Khung Chat Hỏi Đáp AI 💬")
            
            # Khởi tạo lịch sử chat
            if "messages" not in st.session_state:
                st.session_state["messages"] = [
                    {
                        "role": "model", 
                        "content": "Chào bạn! Bạn có thể hỏi tôi bất cứ điều gì về dữ liệu tài chính vừa được tải lên."
                    }
                ]
            
            # Lấy đối tượng chat
            chat_session = setup_gemini_chat(data_for_ai)
            
            if chat_session:
                # Hiển thị lịch sử chat
                for message in st.session_state["messages"]:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
                
                # Xử lý input từ người dùng
                if prompt := st.chat_input("Hỏi Gemini về báo cáo tài chính..."):
                    # 1. Thêm tin nhắn người dùng vào lịch sử
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                        
                    # 2. Gửi tin nhắn đến Gemini và hiển thị phản hồi
                    with st.chat_message("model"):
                        with st.spinner("Đang suy nghĩ..."):
                            try:
                                # Dùng hàm send_message của chat session để duy trì ngữ cảnh
                                response = chat_session.send_message(prompt) 
                                st.markdown(response.text)
                                # 3. Thêm phản hồi của Gemini vào lịch sử
                                st.session_state["messages"].append({"role": "model", "content": response.text})
                            except Exception as e:
                                st.error(f"Lỗi gửi tin nhắn: {e}")
                                st.session_state["messages"].append({"role": "model", "content": "Rất tiếc, đã xảy ra lỗi khi xử lý yêu cầu của bạn."})
            else:
                 st.error("Không thể kích hoạt khung chat. Vui lòng kiểm tra Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets.")


    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}. Vui lòng kiểm tra file excel có đủ 3 cột ('Chỉ tiêu', 'Năm trước', 'Năm sau') và có chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
