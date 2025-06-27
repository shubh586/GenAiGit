# to fetch the transcript and clean it
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
import re
from typing import List


if __name__ == "__main__":
    def extract_video_id(url: str) -> str | None:
        match = re.search(r'(?:v=|\/)([0-9A-Za-z_-]{11})(?=[&?\/]|\s|$)', url)
        return match.group(1) if match else None

    def getRefineTranscript(transcript_list: List[dict]) -> str:
        final_string = ""
        for list1 in transcript_list:
            final_string += list1['text'] + " "
        return final_string.strip()

    def getTranscript(url:str)->List[dict]|None:
        video_id=extract_video_id(url)
        if video_id:
            try:
                transcript_list=YouTubeTranscriptApi.get_transcript(video_id,   languages=["en"])
                final_string=getRefineTranscript(transcript_list)
                return final_string

            except TranscriptsDisabled:
                print("No captions available for this video.")
        else:
            return None









