import pyrebase

firebaseConfig = {
  "apiKey": "AIzaSyCnjgf_SEWR0hWurusvhGfLCP1cYHJ2Ojw",
  "authDomain": "warp-91854.firebaseapp.com",
  "databaseURL": "https://warp-91854-default-rtdb.asia-southeast1.firebasedatabase.app",
  "projectId": "warp-91854",
  "storageBucket": "warp-91854.appspot.com",
  "messagingSenderId": "701313910071",
  "appId": "1:701313910071:web:505da8f99f6b23be99c6b2",
  "measurementId": "G-88B4VPPBB1"
}

firebase = pyrebase.initialize_app(firebaseConfig)

storage = firebase.storage()