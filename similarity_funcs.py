# -*- coding: cp1251 -*-
##
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from tqdm import tqdm


# Функции взяты из документации разработчиков
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_sentence_embedding(title, abstract, model, tokenizer, max_length=None):
    # Tokenize sentences
    sentence = '</s>'.join([title, abstract])
    encoded_input = tokenizer(
        [sentence], padding=True, truncation=True, return_tensors='pt', max_length=max_length).to(model.device)
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings.cpu().detach().numpy()[0]


def get_embedding(all_texts, max_length=1500, verbose=True):
    """ Функция для получения среднего эмбеддинга для всей статьи,
    так как у модели существует ограничение контекста """
    tokenizer = AutoTokenizer.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny")
    model = AutoModel.from_pretrained("mlsa-iai-msu-lab/sci-rus-tiny")
    # model = torch.compile(model)
    model.cuda()
    averages = []
    if verbose:
        for text in tqdm(all_texts, 'Получение эмбеддингов'):
            # Разбиваем текст на части по max_length токенов
            parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            embeddings = []
            for part in parts:
                # Вычисляем эмбендинг для каждой части
                embedding = get_sentence_embedding(part, '', model, tokenizer, max_length)
                embeddings.append(embedding)
            avg_emb = np.mean(embeddings, axis=0)  # Вычисляем средний эмбендинг для всего текста
            averages.append(avg_emb)
        return averages
    else:
        for text in all_texts:
            # Разбиваем текст на части по max_length токенов
            parts = [text[i:i + max_length] for i in range(0, len(text), max_length)]
            embeddings = []
            for part in parts:
                # Вычисляем эмбендинг для каждой части
                embedding = get_sentence_embedding(part, '', model, tokenizer, max_length)
                embeddings.append(embedding)
            avg_emb = np.mean(embeddings, axis=0)  # Вычисляем средний эмбендинг для всего текста
            averages.append(avg_emb)
        return averages


def get_example_narratives():
    used_narratives = [
        "государство разрешает рыночную экономику",
        "государство разрешает свободные цены",
        "государство дерегулирует экономику",
        "государство осуществляет приватизацию убыточных предприятий государственного сектора",
        "государство не вмешивается в рынки",
        "рынки способствуют формированию свободных цен",
        "Частный собственник конкурирует экономически эффективно",
        "государство ограничивает деятельность государство",
        "государство защищает права собственности",
        "государство минимизирует дефицит бюджета",
        "государство снижает расходы на государственное управление и оборону и субсидирование",
        "государство повышает расходы на здравоохранение, образование и инфраструктуру",
        "государство расширяет налоговую базу",
        "государство снижает предельные налоговые ставки",
        "рынки способствуют формированию процентных ставок",
        "государство не вмешивается в формирование процентных ставок",
        "государство поддерживает на низком уровне обменный курс национальной валюты",
        "государство поддерживает единство обменного курса национальной валюты",
        "государство отменяет ограничения экспорта и импорта",
        "государство отменяет ограничения иностранных инвестиций",
        "государство приватизирует государственные предприятия",
        "государство отменяет ограничения конкуренции",
        "государство защищает права собственности",
        "государство приватизирует государственные предприятия",
        "центральный банк осуществляет денежную эмиссию с постоянным низким темпом денежной массы",
        "фирмы производят товары",
        "государство не вмешивается в выбор фирмами производимых товаров",
        "государство управляет для стимулирования развития государственных предприятий",
        "государство концентрирует ресурсы на приоритетных направлениях",
        "государство реализует национальные проекты для экономического роста",
        "государство поддерживает инновационные предприятия",
        "цивилизация развивается экономическая система",
        "государство помогает расти эффективным институтам",
        "государство прикрывается экологической необходимостью",
        "государство управляет экономикой",
        "государство управляет при помощи науки экономикой",
        "государство должно вмешиваться в экономику",
        "государство не вмешивается и проигрывает в экономике",
        "государство сохраняет управление экономикой",
        "государство концентрирует ресурсы на приоритетных направлениях",
        "государство прикрывается экологической необходимостью",
        "цивилизация развивается экономическая система",
        "экономика Русской цивилизации развивается",
        "государство поддерживает инновационные предприятия",
        "государство реализует национальные проекты для экономического роста",
        "государство устраняет имущественное неравенство",
        "государство устраняет неравенство доходов",
        "государство обеспечивает распределительную справедливость",
        "государство устраняет неравенство доходов",
        "государство облагает прогрессивными налогами население",
        "государство обеспечивает социальную защиту населения",
        "государство оказывает адресную поддержку населению",
        "государство выплачивает заработную плату бюджетникам",
        "государство финансирует социальную сферу",
        "государство развивает общественный сектор",
        "государство осуществляет социальную политику для населения",
        "государство строит экономику китайского социализма",
        "государство проводит новую индустриализацию",
        "государство развивает промышленность",
        "государство национализирует предприятия предприятия",
        "государство проводит деприватизацию предприятия"]

    narratives = [
        "государство добивается экономического суверенитета предприятия",
        "государство добивается технологического суверенитета предприятия",
        "предприятия внедряют экологичные технологии",
        "экономические субъекты поддерживают устойчивое развитие",
        "менеджеры бережно относятся к окружающей среде",
        "менеджеры бережно относятся к природе",
        "менеджеры обладают социальной ответственностью",
        "менеджеры обеспечивают высокое качество управления",
        "государство охраняет окружающую среду",
        "государство охраняет природу",
        "государство защищает окружающую среду",
        "государство защищает природу"
    ]

    return narratives


def make_example_tensor(example_triad: list):
    """
    Функция для создания тензора для образцового набора нарративных признаков
    :param example_triad: список строк [актор1, действие1, объект1, актор2, действие2, объект2, ...]
    :return: список torch тензоров
    """

    role_vecs = get_embedding(example_triad)

    example_tensor = torch.tensor(np.stack(role_vecs), dtype=torch.float32)
    return example_tensor


def similarity_economic_meaning(embeddings: list, sample_emb_torch: torch.tensor):
    """
    Функция определяет сходство "экономического смысла" между списком
    нарративных элементов и идеальными образцами
    :param sample_emb_torch: torch тензоры для сравнения (образцовые);
    :param embeddings: список векторов, в котором указаны актор, действие и объект действия;
    :return: значение от 0 до 1, где 1 означает идеальное сходство
    """

    similarities = []
    for idx, emb in enumerate(embeddings):
        # Преобразуем в тензор
        nar_vec_torch = torch.tensor(emb)
        # Считаем сходство
        similarity = util.cos_sim(nar_vec_torch, sample_emb_torch)
        # similarities.append(similarity.item())
        similarities.append(similarity.squeeze().tolist())

    return similarities


def similarity_economic_meaning_addition(embeddings: list, sample_emb_torch: torch.tensor):
    """
    Функция определяет сходство "экономического смысла" между списком
    нарративных элементов и идеальными образцами
    :param sample_emb_torch: torch тензоры для сравнения (образцовые);
    :param embeddings: список векторов, в котором указаны актор, действие и объект действия;
    :return: значение от 0 до 1, где 1 означает идеальное сходство
    """

    similarities = []
    for idx in range(0, len(embeddings), 3):
        # Формируем "ролевое" представление
        # nar_vec = embeddings[idx + 1] + embeddings[idx + 2] - embeddings[idx]  # Действие + Объект - Актор
        nar_vec = embeddings[idx] + embeddings[idx + 1] + embeddings[idx + 2]  # Актор + Действие + Объект

        # Нормализуем вектор
        nar_vec = nar_vec / np.linalg.norm(nar_vec)
        # Преобразуем в тензор
        nar_vec_torch = torch.tensor(nar_vec)
        # Считаем сходство
        similarity = util.cos_sim(nar_vec_torch, sample_emb_torch)
        # similarities.append(similarity.item())
        similarities.append(similarity.squeeze().tolist())

    return similarities


def make_example_tensor_addition(example_triad: list):
    """
    Функция для создания тензора для образцового набора нарративных признаков
    :param example_triad: список строк [актор1, действие1, объект1, актор2, действие2, объект2, ...]
    :return: список torch тензоров
    """

    example_vec = get_embedding(example_triad)
    role_vecs = []

    for idx in range(0, len(example_vec), 3):
        # role_example_vec = example_vec[idx + 1] + example_vec[idx + 2] - example_vec[idx]
        role_example_vec = example_vec[idx] + example_vec[idx + 1] + example_vec[idx + 2]
        role_example_vec = role_example_vec / np.linalg.norm(role_example_vec)
        role_vecs.append(role_example_vec)

    # Преобразуем список numpy-векторов в один torch-тензор
    example_tensor = torch.tensor(np.stack(role_vecs), dtype=torch.float32)
    return example_tensor
