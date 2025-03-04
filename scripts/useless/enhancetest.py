import nlpaug.augmenter.word as naw

# 初始化同义词替换增强器
synonym_aug = naw.SynonymAug(aug_src='wordnet', lang='cmn')  # 'cmn' 表示中文

# 初始化随机删除增强器
random_delete_aug = naw.RandomWordAug()

# 原始句子列表
original_sentences = [
    "人声混响模仿音乐厅",
    "想要后朋克的颓废感",
    "人声更贴耳一点",
    "整体人声可以稍微往后拖一点点，现在听感上有一些前置了，想整体更layback感一些",
    "呼吸声太大了"
]

# 数据增强函数
def augment_text(sentences, augmenters):
    augmented_data = []
    for sentence in sentences:
        for aug in augmenters:
            augmented_sentence = aug.augment(sentence)
            augmented_data.append(augmented_sentence)
    return augmented_data

# 使用同义词替换和随机删除进行增强
augmenters = [synonym_aug, random_delete_aug]
augmented_sentences = augment_text(original_sentences, augmenters)

# 打印增强后的句子
print("原始句子:")
for sentence in original_sentences:
    print(f"- {sentence}")

print("\n增强后的句子:")
for sentence in augmented_sentences:
    print(f"- {sentence}")
    