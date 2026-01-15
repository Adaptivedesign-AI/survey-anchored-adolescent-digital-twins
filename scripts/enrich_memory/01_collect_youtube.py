# -*- coding: utf-8 -*-
import os
import sys
import yaml
import json
import pandas as pd
import yt_dlp
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm
import time

def load_config():
    base_dir = os.getcwd()
    
    secrets_path = os.path.join(base_dir, "configs", "secrets.yaml")
    if not os.path.exists(secrets_path):
        raise FileNotFoundError(f"Secrets file not found at {secrets_path}. Please create it.")
    with open(secrets_path, 'r') as f:
        secrets = yaml.safe_load(f)
    settings_path = os.path.join(base_dir, "configs", "settings.yaml")
    with open(settings_path, 'r') as f:
        settings = yaml.safe_load(f)
        
    return secrets, settings


def extract_video_id(url):
    try:
        parsed_url = urlparse(url)
        if parsed_url.hostname == 'youtu.be':
            return parsed_url.path[1:]
        if parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
            if parsed_url.path == '/watch':
                p = parse_qs(parsed_url.query)
                return p['v'][0]
            if parsed_url.path[:7] == '/embed/':
                return parsed_url.path.split('/')[2]
            if parsed_url.path[:3] == '/v/':
                return parsed_url.path.split('/')[2]
    except Exception as e:
        print(f"Error parsing URL {url}: {e}")
    return None

def fetch_youtube_metadata(youtube_client, video_id):
    try:
        request = youtube_client.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        
        if not response['items']:
            print(f"No metadata found for {video_id}")
            return None
            
        item = response['items'][0]
        metadata = {
            "id": video_id,
            "title": item['snippet']['title'],
            "description": item['snippet']['description'],
            "tags": item['snippet'].get('tags', []),
            "publishedAt": item['snippet']['publishedAt'],
            "channelTitle": item['snippet']['channelTitle'],
            "duration": item['contentDetails']['duration'],
            "viewCount": item['statistics'].get('viewCount', 0),
            "likeCount": item['statistics'].get('likeCount', 0)
        }
        return metadata
    except Exception as e:
        print(f"API Error fetching metadata for {video_id}: {e}")
        return None

def download_audio(video_id, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f'{video_id}.mp3')
 
    if os.path.exists(file_path) and os.path.getsize(file_path) > 1024:
        return True, "Skipped (Already exists)"

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(id)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    
    try:
        url = f"https://www.youtube.com/watch?v={video_id}"
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return True, "Downloaded"
    except Exception as e:
        return False, str(e)


def main():
    print(">>> Step 01: Starting YouTube Collection Pipeline...")
    
    secrets, settings = load_config()
    youtube_api_key = secrets['youtube_api_key']
    
    youtube = build("youtube", "v3", developerKey=youtube_api_key)

    csv_path = settings['paths']['video_csv']
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found at {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} videos from {csv_path}")

    audio_dir = settings['paths']['output_audio_dir']
    meta_dir = settings['paths']['output_metadata_dir']
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)

    results = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        original_link = row.get('link') or row.get('Link')
        csv_title = row.get('video titles') or row.get('Video titles')
        csv_category = row.get('category') or row.get('Category')
        csv_format = row.get('Format')
        
        video_id = extract_video_id(original_link)
        if not video_id:
            print(f"Could not extract ID from {original_link}")
            continue

        success, msg = download_audio(video_id, audio_dir)
        
        meta_file_path = os.path.join(meta_dir, f"{video_id}.json")

        if os.path.exists(meta_file_path):
            with open(meta_file_path, 'r', encoding='utf-8') as f:
                full_metadata = json.load(f)
        else:
            api_metadata = fetch_youtube_metadata(youtube, video_id)
            if api_metadata:
                full_metadata = api_metadata
                full_metadata['manual_label'] = {
                    'category': csv_category,
                    'format': csv_format,
                    'csv_title': csv_title
                }
                with open(meta_file_path, 'w', encoding='utf-8') as f:
                    json.dump(full_metadata, f, ensure_ascii=False, indent=4)
            else:
                full_metadata = None

        results.append({
            "video_id": video_id,
            "download_status": "Success" if success else "Failed",
            "metadata_status": "Success" if full_metadata else "Failed"
        })
        
        time.sleep(0.1)

    df_res = pd.DataFrame(results)
    print("\nProcessing Complete.")
    print(df_res['download_status'].value_counts())
    
if __name__ == "__main__":
    main()