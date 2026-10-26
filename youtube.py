import yt_dlp

url = "https://youtu.be/S8tpzqrwnHk"

options = {
    "format" : "bestaudio/best",
    "outtmpl" : "audio/%(id)s.%(ext)s",
    "quiet" : False,
}

with yt_dlp.YoutubeDL(options) as ydl:
    info = ydl.extract_info(url, download=True)
    
print("Title : ", info["title"])
print("Video id : ", info["id"])
# print("Duration : ", info["duration"], "seconds")