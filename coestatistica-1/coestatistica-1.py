import pandas as pd


def main():
    df = pd.read_csv('desafio1.csv')
    submit = df.groupby('estado_residencia')['pontuacao_credito'].agg([pd.Series.mode, 'median', 'mean', 'std']).T
    submit.rename({'mode': 'moda', 'median': 'mediana', 'mean': 'media', 'std': 'desvio_padrao'})
    submit.to_json('submission.json')
    

if __name__ == '__main__':
    main()

