// config.js
// Environment configuration for CropGuard Pro

let API_BASE_URL;

// Check if we are running on a local computer
if (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1" || window.location.hostname === "") {
    
    // Local Flask Backend
    API_BASE_URL = "http://127.0.0.1:5000";
    console.log("🔧 Running in Development Mode. API:", API_BASE_URL);

} else {
    
    // Live Cloud Backend
    // IMPORTANT: Replace this URL with your actual Render/Railway backend URL once deployed!
    API_BASE_URL = "https://crop-dashboard-w9tt.onrender.com"; 
    console.log("🌍 Running in Production Mode. API:", API_BASE_URL);

}