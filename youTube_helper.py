'''
Transcription utilities that generate textual summaries of Youtube videos, given their URL(s)
- video_get uses pytube to download the video URLs into a local file'''

from pytube import YouTube

def video_get(video_url):
    print(f"videos url --> ", video_url)
    yt = YouTube("https://youtube.com"+ video_url,use_oauth=False, allow_oauth_cache=True)
    print(f"youtube to be downloadd - {yt}")
    vpath = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download()
    print(f"Downloaded video {vpath}")
    return vpath