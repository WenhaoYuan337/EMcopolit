import os,shutil,random
random.seed(0)
from tqdm import tqdm
for i in tqdm(os.listdir('data annotated')):
    os.system('labelme_json to_dataset data annotated/{}'.format(i))

data = [f'data annotated/(i)' for i in os. listdir(r'data annotated') if os.path.isdir(f'data_annotated/{i}')]
random.shuffle(data)
with open('train.txt','w+') as f:
    f.write('\n'.join(data[:int(len(data)*0.8)]))
with open('text.txt','w+') as f:
    f.write('\n'.join(data[:int(len(data)*0.8)]))