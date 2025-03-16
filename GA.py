import random
import time
import uuid
from dataclasses import dataclass

import torch
from torch import GradScaler, autocast
from torch import nn, optim

from dynamic_CNN import generate_random_cp_layer, generate_random_f_layer, DynamicCnn


@dataclass
class Individual:
    id: str
    genes: list
    model: nn.Module


class GA:
    def __init__(self, trainloader, evaloader, population_size, generations=10, mutation_rate=0.1, crossover_rate=0.7):
        self.trainloader = trainloader  # 训练集
        self.evaloader = evaloader  # 评估集
        self.population_size = population_size  # 种群数量,用于初始化和控制后续种群繁殖
        self.fitness_map = {}  # 存储个体id和对应的适应度评分
        self.generations = generations  # 迭代代数
        self.mutation_rate = mutation_rate  # 变异率
        self.crossover_rate = crossover_rate  # 交叉率

        self.population = self.initialize_population(population_size)  # 初始化种群

    # 初始化种群
    def initialize_population(self, size):
        population = {}
        # 随机生成一个大小确定的种群
        for _ in range(size):
            # 随机生成1个个体基因组并创建model
            id, genes = self.random_id_and_genes()
            model = self.create_model(genes)
            # 在创建种群时直接筛除掉一部分明显不适宜的个体
            if model is not None:
                population[id] = Individual(id=id, genes=genes, model=model)

        return population

    # 生成一个包含若干CNN层参数的基因组, 注意此时还未创建CNN
    @staticmethod
    def random_id_and_genes():
        print("正在创建个体")
        genes = list()
        id = str(uuid.uuid4())  # 为每个个体生成唯一ID

        # 添加若干连续的CP层基因
        for i in range(random.randint(2, 5)):
            genes.append(generate_random_cp_layer())

        # 添加若干连续的F层基因
        for i in range(random.randint(0, 2)):
            genes.append(generate_random_f_layer())

        return id, genes

    # 根据个体基因创建CNN模型
    @staticmethod
    def create_model(genes):
        print("正在通过个体基因创建CNN实例")
        cnn = DynamicCnn()
        for gene in genes:
            layer_type, params = gene  # 从基因中解构出 layer_type 和 params
            cnn.add_layer(layer_type, params)

        # 添加最后一个全连接层，设置out_feature为10，不加入基因组
        cnn.add_layer('F', 10)

        # 删除最后一层 ReLU
        if isinstance(cnn.F_layers[-1], nn.ReLU):
            cnn.F_layers = cnn.F_layers[:-1]

        # 去除图像尺寸<=1的无效结构
        if cnn.image_size <= 1:
            return None

        # 初始化模型参数
        cnn.init_weights()
        return cnn

    # 训练模型的函数
    def train(self, model_id, model, epochs=5):
        if epochs == 0:
            return self.fitness_map[model_id]

        trainloader = self.trainloader

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()  # 分类问题的损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

        # 记录最后一次的loss
        last_loss = 0.0

        # 检查是否有可用的 GPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 将模型转移到 GPU
        model.to(device)

        # 创建混合精度缩放器
        scaler = GradScaler()

        # 记录训练开始时间
        start_time = time.time()

        for epoch in range(epochs):
            model.train()  # 切换到训练模式
            running_loss = 0.0

            for inputs, labels in trainloader:
                # 将数据移到 GPU
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                # 混合精度前向传播
                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                # 缩放后的反向传播
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                running_loss += loss.item()

            print(f'模型{model_id}: Epoch {epoch + 1}, Loss: {running_loss / len(trainloader):.4f}')
            last_loss = running_loss / len(trainloader)

        # 计算训练时间
        end_time = time.time()
        training_time = end_time - start_time
        minutes = int(training_time // 60)  # 获取分钟数
        seconds = int(training_time % 60)  # 获取剩余的秒数

        print(f"模型{model_id}训练完毕, 耗时: {minutes} 分 {seconds} 秒")
        return last_loss

    # 评估模型的函数
    def eval(self, model_id, model):
        evaloader = self.evaloader

        # 确保模型在正确的设备上
        device = next(model.parameters()).device  # 获取模型所在的设备

        model.eval()  # 切换到评估模式

        correct = 0
        total = 0
        with torch.no_grad():
            for data in evaloader:
                images, labels = data

                # 将输入数据移到模型所在的设备
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'模型{model_id}评估集准确率: {100 * correct / total:.2f}%')
        return correct / total

    # 计算基因组适应度的函数
    def cal_fitness(self, individual: Individual, generation):
        print(f"正在计算模型{individual.id}基因组适应度\n")
        print(f"模型结构:\n {individual.model}")
        # 通过基因组创建模型
        model = individual.model
        if model is not None:
            # 根据当前迭代次数调整epoch数
            epochs = generation % 3 if individual.id in self.fitness_map.keys() else (2 + generation // 3)
            loss = self.train(individual.id, model, epochs)
            accuracy = self.eval(individual.id, model)
            return accuracy - loss

    # 选择操作, 筛选精英个体作为父代, 清除其余个体
    def select_parents(self, tournament_size=3, elitism=1):
        """
        :param tournament_size: 随机抽取小批量样本进行锦标赛
        :param elitism: 直接从排序的种群中选取前elitism个个体加入新种群
        :return:
        """
        sorted_population = sorted(self.population.values(), key=lambda ind: self.fitness_map[ind.id], reverse=True)

        new_population = {individual.id: individual for individual in sorted_population[:elitism]}  # 直接保留前 `elitism` 个个体
        selected_ids = set(new_population.keys())  # 记录已选中的个体

        # 筛选至原始种群的一半
        while len(new_population) < self.population_size // 2:
            tournament = random.sample([ind for ind in sorted_population if ind.id not in selected_ids], tournament_size)
            best_individual = max(tournament, key=lambda ind: self.fitness_map[ind.id])

            new_population[best_individual.id] = best_individual
            selected_ids.add(best_individual.id)

        # 更新种群
        self.population = new_population

        # 清除未被选择的个体的适应度
        for id in list(self.fitness_map.keys()):
            if id not in selected_ids:
                print(f"淘汰基因组{id}")
                del self.fitness_map[id]

    # 交叉操作
    def crossover(self, individual1, individual2):
        print("交叉操作")
        # 拆分CP基因和F基因
        cp1 = [gene for gene in individual1.genes if gene[0] in ['C', 'P']]
        cp2 = [gene for gene in individual2.genes if gene[0] in ['C', 'P']]

        f1 = [gene for gene in individual1.genes if gene[0] == 'F']
        f2 = [gene for gene in individual2.genes if gene[0] == 'F']

        # CP基因部分的随机片段交叉
        if cp1 and cp2:  # 确保两者均有CP片段,长度至少为1
            cp_min_len = min(len(cp1), len(cp2))

            # 选择截取片段长度,小于等于二者长度最小值
            cut_cp_len = random.randint(1, cp_min_len)

            start1 = random.randint(0, len(cp1) - cut_cp_len)
            start2 = random.randint(0, len(cp2) - cut_cp_len)

            # 生成新 CP 基因
            new_cp_1 = cp1[:start1] + cp2[start2:start2 + cut_cp_len] + cp1[start1 + cut_cp_len:]
            new_cp_2 = cp2[:start2] + cp1[start1:start1 + cut_cp_len] + cp2[start2 + cut_cp_len:]
            new_cp = new_cp_1 if random.random() < 0.5 else new_cp_2

        else:
            return None  # 个体必须有CP基因，没有CP基因直接不生产个体

        # F基因的随机片段交叉
        if f1 and f2:
            f_min_len = min(len(f1), len(f2))

            # 选择截取片段长度,小于等于二者长度最小值
            cut_f_len = random.randint(1, f_min_len)

            start1 = random.randint(0, len(f1) - cut_f_len)
            start2 = random.randint(0, len(f2) - cut_f_len)

            # 生成新F基因
            new_f_1 = f1[:start1] + f2[start2:start2 + cut_f_len] + f1[start1 + cut_f_len:]
            new_f_2 = f2[:start2] + f1[start1:start1 + cut_f_len] + f2[start2 + cut_f_len:]
            new_f = new_f_1 if random.random() < 0.5 else new_f_2
        else:
            new_f = f1 if f1 else f2 if f2 else []

        child_genes = new_cp + new_f
        child_id = str(uuid.uuid4())
        child_model = self.create_model(child_genes)

        return Individual(child_id, child_genes, child_model)

    # 变异操作
    def mutation(self, individual):
        print("变异操作")
        mut_point = random.randint(0, len(individual.genes) - 1)
        gene = individual.genes[mut_point]
        if gene[0] in ['C', 'P']:
            individual.genes[mut_point] = generate_random_cp_layer()
        elif gene[0] == 'F':
            individual.genes[mut_point] = generate_random_f_layer()
        individual.model = self.create_model(individual.genes)

    # 繁殖操作
    def reproduce(self):
        new_population = list(self.population.values())  # 当前种群（已通过选择操作）
        offspring = []  # 用于存储新生成的子代个体

        # 生成子代直到种群总数达到原始数量
        while len(offspring) < self.population_size:
            parent1, parent2 = random.sample(new_population, 2)  # 随机选择两个个体作为父代

            # 交叉操作，生成子个体
            child = self.crossover(parent1, parent2)

            # 检查交叉是否成功（有时基因不符合要求会返回None）
            if child:
                # 随机决定是否变异
                if random.random() < self.mutation_rate:
                    self.mutation(child)

                offspring.append(child)

        # 将新生成的子代加入种群
        for child in offspring:
            self.population[child.id] = child

    # 迭代操作，已初始化种群
    def evolve(self):
        # 迭代若干代
        for generation in range(self.generations):
            print(f"\n -------------- generation {generation} --------------\n")

            # 评估适应度，更新适应度字典
            for individual in list(self.population.values()):  # 使用 list() 创建副本
                if individual and individual.model:
                    fitness = self.cal_fitness(individual, generation)
                    self.fitness_map[individual.id] = fitness
                else:
                    self.population.pop(individual.id)

            # 进行选择操作, 更新self.population
            self.select_parents()

            # 繁殖新一代
            self.reproduce()

        # 在若干代后的种群中选择最优解
        print("迭代完毕, 正在寻找最优解")
        # 评估适应度，更新适应度字典
        for individual in list(self.population.values()):
            if individual and individual.model:
                fitness = self.cal_fitness(individual, self.generations)
                self.fitness_map[individual.id] = fitness
            else:
                self.population.pop(individual.id)

        sorted_population = sorted(self.population.values(), key=lambda ind: self.fitness_map[ind.id], reverse=True)
        print(sorted_population[0])

        torch.save(sorted_population[0].model.state_dict(), 'OPT_model.pth')
