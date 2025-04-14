from collections import Counter, defaultdict
import re
import argparse
import json







def word_stat(text):
    words = re.findall(r"[a-zA-Z0-9']+", text.lower())
    word_counts = Counter(words)
    top_30 = word_counts.most_common(30)
    print(top_30)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', help="full file path",  nargs = '?', const = "bespokelabs_4042025.json", default = "bespokelabs_4042025.json", type= str)
    parser.add_argument('--filenames', help="full file path",  nargs = '?', const = None, default = None, type= str)

    args=parser.parse_args()
    if args.filenames is not None:
        text = []
        for filename in args.filenames.split():
            print(filename)
            with open(filename, 'rb') as f:
                file_raw  = f.read()
                file_json = json.loads(file_raw)
                for item in file_json:
                    text.append(item)
    else:
        with open(args.filename, 'rb') as f:
            cntnt = f.read()
        text = json.loads(cntnt)
    words = []
    tot = len(text)
    lenacc_tot = 0
    lenrej_tot = 0
    for item in text:
        print(len(text))
        rejlen = len(item['rejected'])
        acclen = len(item['accepted'])
        lenrej_tot += rejlen
        lenacc_tot += acclen
        print(f'accepted length {acclen} vs rejected len {rejlen}')
        word = re.findall(r"[a-zA-Z0-9']+", item['rejected'].lower())
        words.extend(word)
    print(f'accepted average length is {lenacc_tot/tot}, rejected average length is {lenrej_tot/tot}')
    word_counts = Counter(words)
    top_30 = word_counts.most_common(30)
    print(top_30)
    combinations = [" ".join(t) for t in zip(*(words[i:] for i in range(2)))]
    combs = Counter(combinations)
    print(combs.most_common(30))
    combinations_3 = [" ".join(t) for t in zip(*(words[i:] for i in range(3)))]
    combs_3 = Counter(combinations_3)
    print(combs.most_common(30))

    while(1):
        word = input("Enter a word: ").split()[0]
        print("You entered:", word)
        print(word_counts[word])
        word =  input("Enter a two word combination: ")
        print("You entered:", word)
        print(combs[word])
        word =  input("Enter a three  word combination: ")
        print("You entered:", word)
        print(combs_3[word])



