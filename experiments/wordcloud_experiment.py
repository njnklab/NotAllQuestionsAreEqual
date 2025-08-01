
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import re
from collections import Counter

# 量表题干映射字典
SCALE_ITEMS = {
    # PHQ-9 题干
    'PHQ1': 'Little interest or pleasure in doing things',
    'PHQ2': 'Feeling down, depressed, or hopeless', 
    'PHQ3': 'Trouble falling or staying asleep, or sleeping too much',
    'PHQ4': 'Feeling tired or having little energy',
    'PHQ5': 'Poor appetite or overeating',
    'PHQ6': 'Feeling bad about yourself or that you are a failure or have let yourself or your family down',
    'PHQ7': 'Trouble concentrating on things, such as reading the newspaper or watching television',
    'PHQ8': 'Moving or speaking so slowly that other people could have noticed. Or the opposite being so fidgety or restless that you have been moving around a lot more than usual',
    'PHQ9': 'Thoughts that you would be better off dead, or of hurting yourself in some way',
    
    # GAD-7 题干
    'GAD1': 'Feeling nervous, anxious or on edge',
    'GAD2': 'Not being able to stop or control worrying', 
    'GAD3': 'Worrying too much about different things',
    'GAD4': 'Trouble relaxing',
    'GAD5': 'Being so restless that it is hard to sit still',
    'GAD6': 'Becoming easily annoyed or irritable',
    'GAD7': 'Feeling afraid as if something awful might happen',
    
    # ISI 题干
    'ISI1': 'Severity of sleep onset difficulties falling asleep',
    'ISI2': 'Severity of sleep maintenance difficulties staying asleep', 
    'ISI3': 'Severity of early morning awakening problems',
    'ISI4': 'Satisfaction dissatisfaction with current sleep pattern',
    'ISI5': 'How noticeable to others do you think your sleep problem is in terms of impairing the quality of your life',
    'ISI6': 'How worried distressed are you about your current sleep problem',
    'ISI7': 'To what extent do you consider your sleep problem to interfere with your daily functioning',
    
    # PSS 题干
    'PSS1': 'How often have you been upset because of something that happened unexpectedly',
    'PSS2': 'How often have you felt that you were unable to control the important things in your life',
    'PSS3': 'How often have you felt nervous and stressed',
    'PSS4': 'How often have you felt confident about your ability to handle your personal problems',
    'PSS5': 'How often have you felt that things were going your way',
    'PSS6': 'How often have you found that you could not cope with all the things that you had to do',
    'PSS7': 'How often have you been able to control irritations in your life',
    'PSS8': 'How often have you felt that you were on top of things',
    'PSS9': 'How often have you been angered because of things that were outside of your control',
    'PSS10': 'How often have you felt difficulties were piling up so high that you could not overcome them',
    'PSS11': 'How often have you been able to control the way you spend your time',
    'PSS12': 'How often have you felt that you were effectively coping with important changes that were occurring in your life',
    'PSS13': 'How often have you felt confident about your ability to handle your personal problems',
    'PSS14': 'How often have you felt that things were going your way'
}

# 停用词列表
STOP_WORDS = {
    'the', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 
    'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she', 'or', 
    'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 
    'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 'no', 'just',
    'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could', 'them', 'see',
    'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think', 'also', 'back',
    'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want',
    'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
    'be', 'were', 'said', 'each', 'did', 'being', 'having', 'such', 'much', 'very', 'more', 'many',
    'often', 'how', 'that', 'things', 'something', 'anything', 'yourself', 'been', 'able', 'felt'
}

def clean_text(text):
    # 转换为小写
    text = text.lower()
    # 移除标点符号，保留字母
    text = re.sub(r'[^\w\s]', ' ', text)
    # 分词
    words = text.split()
    # 移除停用词和短词
    words = [word for word in words if word not in STOP_WORDS and len(word) > 2]
    return ' '.join(words)

def create_unified_wordcloud():
    # 读取排名数据
    df = pd.read_csv('../../results/LIRA/ipca/sample_item_ranks.csv')
    
    # 计算所有项目的平均排名位置
    item_positions = {}
    
    for idx, row in df.iterrows():
        item_ranks = row.iloc[1:].values  # 跳过第一列（样本ID）
        for pos, item in enumerate(item_ranks):
            if pd.notna(item) and item in SCALE_ITEMS:
                if item not in item_positions:
                    item_positions[item] = []
                item_positions[item].append(pos)
    
    # 计算平均位置并转换为权重
    word_weights = {}
    max_pos = max([max(positions) for positions in item_positions.values()]) if item_positions else 0
    
    for item, positions in item_positions.items():
        avg_pos = np.mean(positions)
        # 位置越靠前（数值越小），权重越高
        weight = max_pos - avg_pos + 1
        
        item_text = SCALE_ITEMS[item]
        cleaned_text = clean_text(item_text)
        
        # 将清理后的文本按词分割并加权
        words = cleaned_text.split()
        for word in words:
            if word in word_weights:
                word_weights[word] += weight
            else:
                word_weights[word] = weight
    
    if not word_weights:
        print("没有找到有效的词语权重")
        return
    
    # 创建词云图
    wc = WordCloud(
        width=600, 
        height=150,
        background_color='white',
        colormap='viridis',
        max_words=30,
        relative_scaling=0.5,
        random_state=42
    ).generate_from_frequencies(word_weights)
    
    # 显示和保存词云图
    plt.figure(figsize=(6, 1))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    output_path = '../../results/LIRA/ipca/unified_wordcloud.jpg'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"词云图已保存到: {output_path}")
    print(f"共处理了 {len(df)} 个样本")
    print(f"提取了 {len(word_weights)} 个关键词")
    
    word_freq_df = pd.DataFrame(list(word_weights.items()), 
                              columns=['Word', 'Weight']).sort_values('Weight', ascending=False)
    print("\n前20个重要词汇:")
    for i, (word, weight) in enumerate(word_freq_df.head(20).values, 1):
        print(f"{i:2d}. {word}: {weight:.2f}")

if __name__ == "__main__":
    print("开始生成统一的心理健康量表重要性词云图...")
    create_unified_wordcloud()
    print("词云图生成完成！") 