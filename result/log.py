import pandas as pd
import matplotlib.pyplot as plt
import sys

def main():
    if len(sys.argv) < 2:
        sys.exit('%s [filename]' % sys.argv[0])

    for filename in sys.argv[1:]:
        print filename
        df = pd.read_pickle(filename)
        df['epoch'] = df.index.astype(int)
        df = df.sort('epoch')
        df = df.set_index('epoch')

        plt.figure(figsize=(8,6),dpi=100)

        plt.ylim(0,50)
        plt.ylabel('error rate (%)')
        plt.xlim(0,df.index.max())
        plt.xlabel('epochs (iterations)')
        plt.title(filename.replace('_',' ')[:-4])

        valid = df['valid'].dropna() * 100.
        plt.plot(valid.index, valid, label='validation') 
        test = df['test'].dropna() * 100.
        plt.plot(test.index, test, label='test') 
        plt.legend()
        plt.tight_layout()

        plt.savefig(filename.replace('log','png'))

if __name__ == '__main__':
    main()
