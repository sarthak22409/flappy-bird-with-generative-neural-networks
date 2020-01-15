import pygame
import neat
import time
import os
import random
import pickle
pygame.font.init()  # init font

#from pygame.examples.video import win

WIN_WIDTH = 600
WIN_HEIGTH = 800
STAT_FONT = pygame.font.SysFont("comicsans", 50)

BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird1.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird2.png"))),pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird3.png")))]  #loading images
PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bg.png")))

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROTATION =25 # to control rotation so that we don't change it too much
    ROT_VEL =20
    ANIMATION_TIME = 5  #by changing this we can change how fast birds gonna flap their wings


    def __init__(self,x,y):
        self.x = x
        self.y=y
        self.tilt =0
        self.tick_count = 0
        self.vel =0
        self.height =self.y
        self.img_count = 0
        self.img = self.IMGS[0]

    def jump(self):
        self.vel = -10.5
        self.tick_count = 0 #to keep  in account when we last jumped
        self.height = self.y

    def move(self):
        self.tick_count+=1 #a frame went by means last time we jumped
        d = self.vel*self.tick_count + 1.5*self.tick_count**2 # results in arc for a bird
        if d>=16:
            d =16
        if d<0:
            d-=2
        self.y = self.y+d # change in y poistion
        if d<0 or self.y < self.height+50:  #if d is less than zero that means we re moving upwards thus pointing bird's beak upwards and also pointing it upwards untill it reaches top most point
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION

        else:
            if self.tilt>-90:
                self.tilt-=self.ROT_VEL

    def draw(self,win):
        self.img_count+=1

        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count < self.ANIMATION_TIME*4: #extra 2 elifs statement are added so that our bird bring down wings smoothly
            self.img = self.IMGS[0]
        elif self.img_count < self.ANIMATION_TIME*4+1:
            self.img = self.IMGS[0]
            self.img_count = 0

        if self.tilt<=-80: #if bird is goingdown it should not be flapping its wings
            self.img = self.IMGS[1]  #non flapping bird imagges
            self.img_count = self.ANIMATION_TIME*2

        rotated_image = pygame.transform.rotate(self.img,self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft = (self.x,self.y)).center)  #rotates image about center
        win.blit(rotated_image,new_rect.topleft)
    def get_msk(self):
        return pygame.mask.from_surface(self.img)  # for collision with object


class pipe:
    GAP = 200 #gap between the pipes
    VEL =5
    def __init__(self,x):
        self.x =x
        self.height = 0


        self.top = 0  #to keep in account no of pipes for top
        self.bottom= 0 #to keep in account no of pipes for top
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG,False,True)
        self.PIPE_BOTTOM = PIPE_IMG
        self.passed = False
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        self.x-=self.VEL


    def draw(self,win):
        win.blit(self.PIPE_TOP,(self.x,self.top))
        win.blit(self.PIPE_BOTTOM,(self.x,self.bottom))
    def collide(self,bird):
        bird_mask = bird.get_msk()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)  #give mask(2D array) area
        bottom_mask = pygame.mask.from_surface((self.PIPE_BOTTOM)) #give mask(2D array) area
        top_offset = (self.x-bird.x,self.top-round(bird.y))
        bottom_offset = (self.x-bird.x,self.bottom-round(bird.y))

        b_point = bird_mask.overlap(bottom_mask,bottom_offset)
        t_point = bird_mask.overlap(top_mask,top_offset)
        if t_point or b_point:
            return True
        return False

class Base:
    VEL =5
    WIDTH = BASE_IMG.get_width()
    IMG = BASE_IMG

    def __init__(self,y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        self.x1-=self.VEL
        self.x2 -=self.VEL
        if self.x1 +self.WIDTH < 0: #condition to check if any one is complete of the screen
            self.x1 = self.x2+self.WIDTH
        if self.x2+self.WIDTH < 0:
            self.x2 = self.x1+self.WIDTH

    def draw(self,win):
        win.blit(self.IMG,(self.x1,self.y))
        win.blit(self.IMG,(self.x2,self.y))


def draw_window(win,birds,pipes,base,score):
    win.blit(BG_IMG,(0,0)) #draw whatever you want todraw on your windows
    for pipe in pipes:
        pipe.draw(win)
    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))
    base.draw(win)
    for bird in birds:
        bird.draw(win)
    pygame.display.update()


def main(genomes,config):
    nets = []  #to keep track of neural network
    ge = []  #to keep track of bid which it is controlling
    birds = []
    for _,g in genomes: #genome is a tuple
        net = neat.nn.FeedForwardNetwork.create(g,config)
        nets.append(net)
        birds.append(Bird(230,350))
        g.fitness = 0
        ge.append(g)


    base = Base(730)
    pipes=[pipe(700)]  #at max two pipes in a list
    win = pygame.display.set_mode((WIN_WIDTH,WIN_HEIGTH))
    clock = pygame.time.Clock()
    run =True
    score =0

    while run: #event loop
        clock.tick(30)
        for event in pygame.event.get():  # keep track of events
            if event.type == pygame.quit:
                run = False
                pygame.quit()
                quit()
        pipe_ind =0
        if len(birds) >0:
            if len(pipes)>1 and birds[0].x +pipes[0].x +pipes[0].PIPE_TOP.get_width():  #this tells us if the bird has already crossed the pipe
                pipe_ind = 1  #the  we are changing pipe_ind to 2nd pipe
        else:
            run = False
            break
        for x,bird in enumerate(birds):
            bird.move()
            ge[x].fitness +=0.1

            output = nets[x].activate((bird.y,abs(bird.y-pipes[pipe_ind].height),abs(bird.y-pipes[pipe_ind].bottom)))
            if output[0]>0.5:
                bird.jump()
        rem = []
        add_pipe= False
        for piPe in pipes:
            for x,bird in enumerate(birds):
                if piPe.collide(bird):
                    ge[x].fitness-=1
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)


                if not piPe.passed and piPe.x<bird.x:
                    piPe.passed=True
                    add_pipe = True
            if piPe.x + piPe.PIPE_TOP.get_width()<0:  #only  for one pipe
                rem.append(piPe)
            piPe.move()

        if add_pipe:
            score+=1
            for g in ge:
                g.fitness+=5   # we are increasing fitness by 5 we are putting emphasis on getting through the pipe
            pipes.append(pipe(600))
        for r in rem:
            pipes.remove(r)

        for x,bird in enumerate(birds):
            if bird.y+bird.img.get_height()>=730 or bird.y<0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)

        base.move()
        draw_window(win,birds,pipes,base,score)



def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    p = neat.Population(config)
    p.add_reporter((neat.StdOutReporter(True)))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    winner = p.run(main,50)  #we detremine fitness function as how far our bird is going to move


if __name__=="__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir,"config.txt")
    run(config_path)