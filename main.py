
from environment_setup import setup_environment
from model_loader import load_model
from song_generator import generate_song
from output_handler import save_output

def main():
    """Main pipeline to generate a Jukebox song."""
    setup_environment()
    model = load_model("1b_lyrics")
    output_dir = save_output("./outputs")
    generate_song("1b_lyrics", 30, output_dir, "sample_30s")

if __name__ == "__main__":
    main()
