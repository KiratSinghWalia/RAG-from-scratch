import argparse
from lib.multimodel_search import verify_emb,image_search_command


def main():
    parser = argparse.ArgumentParser(description="multi model search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    search_parser = subparsers.add_parser("verify_image_embedding", help="verify the model loaded")
    search_parser.add_argument('filepath',type = str , help='image to embed')

    image_search = subparsers.add_parser("image_search", help="return similar movies")
    image_search.add_argument('filepath',type = str , help='image to search')


    args = parser.parse_args()
    match args.command:
        case 'verify_image_embedding':
            verify_emb(args.filepath)
          
        case 'image_search':
            image_search_command(args.filepath)

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()