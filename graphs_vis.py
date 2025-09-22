##
from pyvis.network import Network
import pandas as pd

data = pd.read_excel(r"E:\Downloads\Триады_нарративов_new2.xlsx")

net = Network(directed=False)

net.add_node(1, label="Актор", title="Актор", shape='circle', color="#22b512")
net.add_node(2, label="Действие", title="Действие", shape="box", color="#b51222")
net.add_node(3, label="Объект", title="Объект", shape="box", color="#42adf5")
# net.add_node(4, label='Модальность', title="Модальность", shape="box", color="#b51222")
# net.add_node(5, label="Описание действия", title="Описание действия", shape="box", color="#b51222")

net.add_edge(1, 2, width=2, arrows="to")
net.add_edge(2, 3, width=2, arrows="to", label='tonality | modality')
# net.add_edge(3, 4, width=1, label='tonality | modality')
# net.add_edge(4, 5, width=1)

net.save_graph('graph.html')

##
counts = data['PER'].value_counts()
print(counts)
##
negative = data[data['tonality'] == 'positive']
print(negative)

##
from pyvis.network import Network
import pandas as pd

# Загрузка данных
data = pd.read_excel(r"E:\Downloads\Триады_нарративов_new2.xlsx")

net = Network(directed=True)  # можно directed=True для стрелок
node_counts = {}  # словарь для подсчёта вхождений узлов

# Цвета и формы по типу
node_styles = {
    'speakers': {'shape': 'box', 'color': '#b56f16'},
    'sentence': {'shape': 'box', 'color': '#6a6561'},
    'actor': {'shape': 'box', 'color': '#22b512'},
    'action': {'shape': 'box', 'color': '#b51222'},
    'object': {'shape': 'box', 'color': '#42adf5'},
    'modality': {'shape': 'box', 'color': '#7b7e7a'},
    'tonality': {'shape': 'box', 'color': '#59939f'},
    'per': {'shape': 'box', 'color': '##da621e'},
    'loc': {'shape': 'box', 'color': '#23d0d5'},
    'org': {'shape': 'box', 'color': '#611ddb'}
}

# Добавление узлов и рёбер
for idx, row in data[:620].iterrows():
    # Получаем значения
    speakers = str(row['speakers'])
    sentences = str(row['sentence'])
    actors = str(row['actors'])
    actions = str(row['actions'])
    objects = str(row['objects'])
    modality = str(row['modality'])
    tonality = str(row['tonality'])
    PERs = str(row['PER'])
    LOCs = str(row['LOC'])
    ORGs = str(row['ORG'])

    # --- Узлы ---
    for label, node_type in zip([speakers, sentences, actors, actions, objects, modality, tonality],
                                ['speakers', 'sentence', 'actor', 'action', 'object', 'modality', 'tonality']):
        # label = label.replace('[', ' ').replace(']', ' ')
        # label = label.replace(',', ' ')
        # if label == "Нейтральная" or label == 'neutral':
        #    continue
        if label == '[]' or label == '-':
            continue
        # if node_type == 'modality' or node_type == 'tonality' or node_type == 'sentence':
        if node_type == 'modality' or node_type == 'tonality':
            continue
        if label in node_counts and node_type == 'sentenc':
            node_counts[label] += 1
            display_label = f"{label} ({node_counts[label]})"
        else:
            node_counts[label] = 1
            display_label = label
            if node_type != 'sentence':
                net.add_node(label, label=display_label, title=label,
                             **node_styles[node_type])
            else:
                net.add_node(label, label='sent', title=display_label,
                             **node_styles[node_type])

    # --- Рёбра ---
    # Актор ? Действие
    net.add_edge(speakers, sentences, width=2, arrows='to')
    try:
        net.add_edge(sentences, actors, width=2, arrows='to')
    except:
        net.add_edge(sentences, actions, width=2, arrows='to')
        pass
    try:
        net.add_edge(actors, actions, width=2, arrows='to')
    except:
        pass
    try:
        # Действие ? Объект
        net.add_edge(actions, objects, width=2, arrows='to', label=f'{modality} | {tonality}')
    except:
        pass
    # Действие ? Модальность
    # net.add_edge(objects, modality, width=1)
    # Действие ? Тональность/Описание действия
    # net.add_edge(modality, tonality, width=1)
    # net.add_edge(tonality, PERs, width=1)
    # net.add_edge(tonality, LOCs, width=1)
    # net.add_edge(tonality, ORGs, width=1)

# Сохранение графа
net.save_graph('graph.html')
##
print(node_counts['[есть]'])


##
