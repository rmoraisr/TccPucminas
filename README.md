# Modelo Supervisionado de Classificação de Sentimentos de Texto

Trabalho de Conclusão de Curso apresentado ao Curso de Especialização em
Ciência de Dados e Big Data como requisito parcial à obtenção do título de
especialista.

![](https://github.com/rmoraisr/TccPucminas/blob/master/scripts/13_TCC/assets/fig/canvas_presentation2.jpeg)

* Apresentação baseada no Data Science Workflow Canvas proposto por [Jasmine Vasandani
](https://towardsdatascience.com/a-data-science-workflow-canvas-to-kickstart-your-projects-db62556be4d0).

## Opção para execução dos Jupyter Notebooks

Opção experimental que dispensa fazer qualquer instalação na máquina local, pode ser executada a partir do repositório [Binder](https://mybinder.org/v2/gh/rmoraisr/TccPucminas/HEAD).

Na primeira execução pode ser necessário executar os comandos da célula reproduzida a seguir, nenhuma instalação será feita localmente. Essas ocorrerão no servidor do repostiório Binder:

```python
# Pacotes necessários (tirar o comentário para instalar)
!pip install pandas
!pip install matplotlib
!pip install sklearn
!pip install nltk
!pip install lime
!pip install seaborn
!pip install matplotlib_venn
```
Caso opte por baixar os jupyter notebooks para rodar localmente, a estrura de arquivos da figura a seguir deve ser respeitada.

<center>
  <img src="https://github.com/rmoraisr/TccPucminas/blob/master/scripts/13_TCC/assets/fig/folders.png" alt="Estrutura de Arquivos" width="250"/>
</center>

Caso deseje alterar o nome das pastas, altere também o trecho de código seguinte:

```python
# Pasta do projeto
folder_path = '../13_TCC/'
```

