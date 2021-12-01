import glob
import random
import xml.etree.ElementTree as ET

def get_persuasive_pairs_xml(directory: str = './16k_persuasiveness/data/UKPConvArg1Strict-XML/', include_metadata: bool = False):
    '''
    Extracts and compiles the 16k persuasiveness pairs from a directory containing XML files

    Args:
        directory (str): directory to 16k persuasive pairs folder with XML files
        include_metadata (bool): whether to include filename in data
    Returns:
        a list of dictionaries with both sentences and the label
    '''
    persuasive_pairs_df = []

    for filename in glob.glob(directory + '*.xml'):
        root = ET.parse(filename).getroot()

        argument_pairs = [type_tag for type_tag in root.findall(
            'annotatedArgumentPair')]

        for argument_pair in argument_pairs:
            sentence_a = argument_pair.find('arg1/text').text
            sentence_b = argument_pair.find('arg2/text').text
            title = argument_pair.find('debateMetaData/title').text
            try:
                description = argument_pair.find('debateMetaData/description').text
            except AttributeError:
                description = title
            
            description = title if description is None else description

            labels = [type_tag.find('value').text for type_tag in argument_pair.findall(
                'mTurkAssignments/mTurkAssignment')]
            labels = filter(lambda x: x != 'equal', labels) 
            labels = list(map(lambda x: int(x[-1]) - 1, labels))
            label = max(labels, key=labels.count)

            persuasive_score = labels.count(label)/len(labels)

            row = {
                    'label': label,
                    'sentence_a': sentence_a.strip().replace('\n', ' '),
                    'sentence_b': sentence_b.strip().replace('\n', ' '),
                    'title': title.strip(),
                    'description': description.strip().replace('\n', ' '),
                    'persuasive_score': persuasive_score
            }

            if include_metadata:
                row['filename'] = filename
            persuasive_pairs_df.append(row)

    return persuasive_pairs_df

def write_persuasive_df_to_txt(persuasive_pairs_df, txt_filename = './persuasive_pairs_data', train_test_split=0.8):

    total_length = len(persuasive_pairs_df)
    random.shuffle(persuasive_pairs_df)
    split = int(train_test_split * total_length)
    train_dataset = persuasive_pairs_df[:split]
    test_dataset = persuasive_pairs_df[split:]

    for dataset, set_name in zip([train_dataset, test_dataset], ['train', 'test']):
        with open(f'{txt_filename}_{set_name}.txt', 'w') as f:
            for row in dataset:
                persuasive_sentence = row['sentence_a'] if not row['label'] else row['sentence_b']
                data_str = row['description'] + '\t' + str(row['persuasive_score']) + '\t' + persuasive_sentence + '\n'
                f.write(data_str)


if __name__ == '__main__':
    persuasive_pairs_df = get_persuasive_pairs_xml('../persuasive_classifier/16k_persuasiveness/data/UKPConvArg1Strict-XML/')
    write_persuasive_df_to_txt(persuasive_pairs_df)