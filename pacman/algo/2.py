from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from util import nearestPoint
from game import Directions
import game
from util import nearestPoint
from game import Actions
import copy

def createTeam(firstIndex, secondIndex, isRed,first = 'AstarAttacker', second = 'AstarDefender'):
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

class ScanMap:
    def __init__(self, gameState, agent):
        self.food = agent.getFood(gameState).asList()
        self.walls = gameState.getWalls()
        self.homeBoundary = agent.boundaryPosition(gameState)
        self.height = gameState.data.layout.height
        self.width = gameState.data.layout.width

    def getFoodList(self, gameState):
        foods = []
        for food in self.food:
            food_fringes = []
            food_valid_fringes = []
            count = 0
            food_fringes.append((food[0] + 1, food[1]))
            food_fringes.append((food[0] - 1, food[1]))
            food_fringes.append((food[0], food[1] + 1))
            food_fringes.append((food[0], food[1] - 1))
            for food_fringe in food_fringes:
                if not gameState.hasWall(food_fringe[0], food_fringe[1]):
                    count = count + 1
                    food_valid_fringes.append(food_fringe)
            if count > 1:
                foods.append((food, food_valid_fringes))
        return foods

    def getSafeFoods(self, foods):
        safe_foods = []
        for food in foods:
            count = self.getNumOfValidActions(food)
            if count > 1:
                safe_foods.append(food[0])
        return safe_foods

    def getDangerFoods(self, safe_foods):
        danger_foods = []
        for food in self.food:
            if food not in safe_foods:
                danger_foods.append(food)
        return danger_foods

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                successors.append((nextState, action))
        return successors

    def isGoalState(self, state):
        return state in self.homeBoundary

    def getNumOfValidActions(self, foods):
        food = foods[0]
        food_fringes = foods[1]
        visited = []
        visited.append(food)
        count = 0
        for food_fringe in food_fringes:
            closed = copy.deepcopy(visited)
            if self.BFS(food_fringe, closed):
                count = count + 1
        return count

    def BFS(self, food_fringe, closed):
        from util import Queue

        fringe = Queue()
        fringe.push((food_fringe, []))
        while not fringe.isEmpty():
            state, actions = fringe.pop()
            closed.append(state)
            if self.isGoalState(state):
                return True
            for successor, direction in self.getSuccessors(state):
                if successor not in closed:
                    closed.append(successor)
                    fringe.push((successor, actions + [direction]))

class DummyAgent(CaptureAgent):
  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)
    self.midWidth = gameState.data.layout.width // 2
    self.height = gameState.data.layout.height
    self.width = gameState.data.layout.width
    self.midHeight = gameState.data.layout.height // 2
    self.foodEaten = 0
    self.initialnumberOfFood = len(self.getFood(gameState).asList())
    self.lastEatenFoodPosition = None
    self.initialnumberOfCapsule = len(self.getCapsules(gameState))
    scanmap = ScanMap(gameState, self)
    foodList = scanmap.getFoodList(gameState)

    self.safeFoods = scanmap.getSafeFoods(foodList) 
    self.dangerFoods = scanmap.getDangerFoods(self.safeFoods)
    self.blueRebornHeight = self.height -1
    self.blueRebornWidth = self.width -1

  def evaluate(self, gameState, action):
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = self.getScore(successor)
    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 1.0}

  def getSuccessor(self, gameState, action):
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def chooseAction(self, gameState):
    self.locationOfLastEatenFood(gameState)  
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def distToFood(self, gameState):
    food = self.getFood(gameState).asList()
    if len(food) > 0:
      dist = 9999
      for a in food:
        tempDist = self.getMazeDistance(gameState.getAgentPosition(self.index), a)
        if tempDist < dist:
          dist = tempDist
          temp = a
      return dist
    else:
      return 0

  def distToHome(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    boundaries = []
    if self.red:
      i = self.midWidth - 1
    else:
      i = self.midWidth + 1
    boudaries = [(i,j) for j in  range(self.height)]
    validPositions = []
    for i in boudaries:
      if not gameState.hasWall(i[0],i[1]):
        validPositions.append(i)
    dist = 9999
    for validPosition in validPositions:
      tempDist =  self.getMazeDistance(validPosition,myPosition)
      if tempDist < dist:
        dist = tempDist
        temp = validPosition
    return dist


  def boundaryPosition(self,gameState):
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    boundaries = []
    if self.red:
      i = self.midWidth - 1
    else:
      i = self.midWidth + 1
    boudaries = [(i,j) for j in  range(self.height)]
    validPositions = []
    for i in boudaries:
      if not gameState.hasWall(i[0],i[1]):
        validPositions.append(i)
    return validPositions


  def distToCapsule(self,gameState):
    if len(self.getCapsules(gameState)) > 1:
      dist = 9999
      for i in self.getCapsules(gameState):
        tempDist = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), i)
        if tempDist < dist:
          dist = tempDist
          self.debugDraw(i, [125, 125, 211], True)
      return dist

    elif len(self.getCapsules(gameState)) == 1 :
      distToCapsule = self.getMazeDistance(gameState.getAgentState(self.index).getPosition(), self.getCapsules(gameState)[0])
      self.debugDraw(self.getCapsules(gameState)[0], [125, 125, 211], True)
      return distToCapsule


  def locationOfLastEatenFood(self,gameState):
    if len(self.observationHistory) > 1:
      prevState = self.getPreviousObservation()
      prevFoodList = self.getFoodYouAreDefending(prevState).asList()
      currentFoodList = self.getFoodYouAreDefending(gameState).asList()
      if len(prevFoodList) != len(currentFoodList):
        for food in prevFoodList:
          if food not in currentFoodList:
            self.lastEatenFoodPosition = food


  def getNearestGhostDistance(self, gameState):
    myPosition =  gameState.getAgentState(self.index).getPosition()
    enemies =  [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in ghosts]
      return min(dists)
    else:
      return None


  def getNearestinvader(self, gameState):
    myPosition =  gameState.getAgentState(self.index).getPosition()
    enemies =  [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPosition, a.getPosition()) for a in invaders]
      return min(dists)
    else:
      return None



  def DistToGhost(self, gameState):
    myPosition =  gameState.getAgentState(self.index).getPosition()
    enemies =  [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    ghosts = [a for a in enemies if not a.isPacman and a.getPosition() != None]
    if len(ghosts) > 0:
      dist = 999
      for a in ghosts:
        temp = self.getMazeDistance(myPosition, a.getPosition())
        if temp < dist:
          dist = temp
          ghostState = a
      return [dist,ghostState]
    else:
      return None


  def opponentscaredTime(self,gameState):
    opponents = self.getOpponents(gameState)
    for opponent in opponents:
      if gameState.getAgentState(opponent).scaredTimer > 1:
        return gameState.getAgentState(opponent).scaredTimer
    return 0

  def nullHeuristic(self,state, problem=None):
    return 0

  def aStarSearch(self, problem, gameState, heuristic=nullHeuristic):
    from util import PriorityQueue
    start_state = problem.getStartState()
    fringe = PriorityQueue()
    h = heuristic(start_state, gameState)
    g = 0
    f = g + h
    start_node = (start_state, [], g)
    fringe.push(start_node, f)
    explored = []
    while not fringe.isEmpty():
      current_node = fringe.pop()
      state = current_node[0]
      path = current_node[1]
      current_cost = current_node[2]
      if state not in explored:
        explored.append(state)
        if problem.isGoalState(state):
          return path
        successors = problem.getSuccessors(state)
        for successor in successors:
          current_path = list(path)
          successor_state = successor[0]
          move = successor[1]
          g = successor[2] + current_cost
          h = heuristic(successor_state, gameState)
          if successor_state not in explored:
            current_path.append(move)
            f = g + h
            successor_node = (successor_state, current_path, g)
            fringe.push(successor_node, f)
    return []



  def GeneralHeuristic(self, state, gameState):
    heuristic = 0
    if self.getNearestGhostDistance(gameState) != None :
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      ghosts = [a for a in enemies if not a.isPacman and a.scaredTimer < 2 and a.getPosition() != None]
      if ghosts != None and len(ghosts) > 0:
        ghostpositions = [ghost.getPosition() for ghost in ghosts]
        ghostDists = [self.getMazeDistance(state,ghostposition) for ghostposition in ghostpositions]
        ghostDist = min(ghostDists)
        if ghostDist < 2:
          heuristic = pow((5-ghostDist),5)
    return heuristic

  def avoidPacmanHeuristic(self, state, gameState):
    weight = 0
    if self.getNearestinvader(gameState) != None and gameState.getAgentState(self.index).scaredTimer > 0:
      enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
      pacmans = [a for a in enemies if a.isPacman and a.getPosition() != None]
      if pacmans != None and len(pacmans) > 0:
        pacmanPositions = [pacman.getPosition() for pacman in pacmans]
        pacmanDists = [self.getMazeDistance(state,pacmanposition) for pacmanposition in pacmanPositions]
        pacmanDist = min(pacmanDists)
        if pacmanDist < 2:
          weight = pow((5- pacmanDist),5)

    return weight
  
class AstarAttacker(DummyAgent):
  def getGhostDistance(self,gameState,index):
    myPosition = gameState.getAgentState(self.index).getPosition()
    ghost = gameState.getAgentState(index)
    dist = self.getMazeDistance(myPosition,ghost.getPosition())
    return dist

  def chooseAction(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    newSafeFoods =[]
    newDangerousFood =[]

    for i in self.getFood(gameState).asList():
      if i in self.safeFoods:
        newSafeFoods.append(i)

    for i in self.getFood(gameState).asList():
      if i in self.dangerFoods:
        newDangerousFood.append(i)
    self.safeFoods = copy.deepcopy(newSafeFoods)
    self.dangerFoods = copy.deepcopy(newDangerousFood)

    if gameState.getAgentState(self.index).numCarrying == 0 and len(self.getFood(gameState).asList()) == 0:
      return 'Stop'

    if len(self.safeFoods) < 1 and len(self.getCapsules(gameState)) != 0 and self.opponentscaredTime(gameState) < 10:
      problem = SearchCapsule(gameState, self, self.index)
      return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]


    if gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) > 0):
      problem = SearchSafeFood(gameState, self, self.index)
      return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if gameState.getAgentState(self.index).numCarrying < 1 and (len(self.safeFoods) == 0):
      problem = SearchFood(gameState, self, self.index)
      return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if self.DistToGhost(gameState) != None and self.DistToGhost(gameState)[0]< 6 and \
        self.DistToGhost(gameState)[1].scaredTimer < 5:
      problem = Escape(gameState, self, self.index)
      if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
        return 'Stop'
      else:
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if self.opponentscaredTime(gameState) != None:
      if self.opponentscaredTime(gameState) > 20 and len(self.dangerFoods) > 0:
        problem = SearchDangerousFood(gameState, self, self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    if len(self.getFood(gameState).asList()) < 3 or gameState.data.timeleft < self.distToHome(gameState) + 60\
        or gameState.getAgentState(self.index).numCarrying > 15:
      problem = BackHome(gameState, self, self.index)
      if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
        return 'Stop'
      else:
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    problem = SearchFood(gameState, self, self.index)
    return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

class PositionSearchProblem:
    def __init__(self, gameState, agent, agentIndex = 0,costFn = lambda x: 1):
        self.walls = gameState.getWalls()
        self.costFn = costFn
        self.startState = gameState.getAgentState(agentIndex).getPosition()
        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 

    def getStartState(self):
      return self.startState

    def isGoalState(self, state):

      util.raiseNotDefined()

    def getSuccessors(self, state):
        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        self._expanded += 1 
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class SearchFood(PositionSearchProblem):
  def __init__(self, gameState, agent, agentIndex = 0):
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0 
    self.carry = gameState.getAgentState(agentIndex).numCarrying
    self.foodLeft = len(self.food.asList())


  def isGoalState(self, state):
    return state in self.food.asList()

class SearchSafeFood(PositionSearchProblem):
  def __init__(self, gameState, agent, agentIndex = 0):
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0 
    self.carry = gameState.getAgentState(agentIndex).numCarrying
    self.foodLeft = len(self.food.asList())
    self.safeFood = agent.safeFoods


  def isGoalState(self, state):
    return state in self.safeFood

class SearchDangerousFood(PositionSearchProblem):
  def __init__(self, gameState, agent, agentIndex = 0):
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  
    self.carry = gameState.getAgentState(agentIndex).numCarrying
    self.foodLeft = len(self.food.asList())
    self.dangerousFood = agent.dangerFoods

  def isGoalState(self, state):
    return state in self.dangerousFood

class Escape(PositionSearchProblem):
  def __init__(self, gameState, agent, agentIndex=0):
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0 
    self.homeBoundary = agent.boundaryPosition(gameState)
    self.safeFood = agent.safeFoods

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    return state in self.homeBoundary or state in self.capsule

class BackHome(PositionSearchProblem):
  def __init__(self, gameState, agent, agentIndex=0):
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  
    self.homeBoundary = agent.boundaryPosition(gameState)

  def getStartState(self):
    return self.startState

  def isGoalState(self, state):
    return state in self.homeBoundary

class SearchCapsule(PositionSearchProblem):
  def __init__(self, gameState, agent, agentIndex = 0):
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  


  def isGoalState(self, state):
    return state in self.capsule

class SearchInvaders(PositionSearchProblem):
    def __init__(self, gameState, agent, agentIndex=0):
        super().__init__(gameState, agent, agentIndex)
        self.enemies = [gameState.getAgentState(i) for i in agent.getOpponents(gameState)]
        self.invaders = [a for a in self.enemies if a.isPacman and a.getPosition() is not None]
        if len(self.invaders) > 0:
            self.invaderPositions = [invader.getPosition() for invader in self.invaders]
        else:
            self.invaderPositions = None

    def isGoalState(self, state):
        return state in self.invaderPositions
    
class SearchLastEatenFood(PositionSearchProblem):
  def __init__(self, gameState, agent, agentIndex = 0):
    self.food = agent.getFood(gameState)
    self.capsule = agent.getCapsules(gameState)
    self.startState = gameState.getAgentState(agentIndex).getPosition()
    self.walls = gameState.getWalls()
    self.lastEatenFood = agent.lastEatenFoodPosition
    self.costFn = lambda x: 1
    self._visited, self._visitedlist, self._expanded = {}, [], 0  

  def isGoalState(self, state):
    return state == self.lastEatenFood

class DefensiveProblem(PositionSearchProblem):
    def __init__(self, gameState, agent, agentIndex=0):
        super().__init__(gameState, agent, agentIndex)
        self.invaders = [
            a for a in agent.getOpponents(gameState) if gameState.getAgentState(a).isPacman
        ]
        self.gameState = gameState  # Add gameState as an attribute

    def isGoalState(self, state):
        if self.invaders:
            # 优先追捕携带食物较多的敌人
            invadersWithFood = [
                (invader, self.gameState.getAgentState(invader).numCarrying)  # Use self.gameState
                for invader in self.invaders
                if self.gameState.getAgentState(invader).numCarrying > 0  # Use self.gameState
            ]
            if invadersWithFood:
                closestInvader, _ = min(
                    invadersWithFood,
                    key=lambda x: self.agent.getMazeDistance(state, x[0].getPosition()),
                )
                return state == closestInvader.getPosition()
            else:
                # 如果没有携带食物的敌人，则追捕最近的敌人
                invaderPositions = [
                    self.gameState.getAgentPosition(invader) for invader in self.invaders  # Use self.gameState
                ]
                closestInvader = min(
                    invaderPositions,
                    key=lambda pos: self.agent.getMazeDistance(state, pos),
                )
                return state == closestInvader
        else:  
            # 否则，目标是最后被吃掉的食物位置或巡逻位置
            return (
                state == self.agent.lastEatenFoodPosition
                or state in self.agent.getPatrolPositions(self.gameState)  # Use self.gameState
            )
        
class AstarDefender(DummyAgent):
  def chooseAction(self, gameState):
    myState = gameState.getAgentState(self.index)
    myPosition = myState.getPosition()
    self.locationOfLastEatenFood(gameState)
    actions = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in actions]
    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman]
    knowninvaders = [a for a in enemies if a.isPacman and a.getPosition() !=None ]

    if len(invaders) == 0 or gameState.getAgentPosition(self.index) == self.lastEatenFoodPosition or len(knowninvaders) > 0:
      self.lastEatenFoodPosition = None
    if len(invaders) < 1:
      if gameState.getAgentState(self.index).numCarrying < 3 and len(self.getFood(gameState).asList()) != 0 and not (self.DistToGhost(gameState) != None and self.DistToGhost(gameState)[0]< 4 and \
        self.DistToGhost(gameState)[1].scaredTimer < 2):
        problem = SearchFood(gameState, self, self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
      else:
        problem = BackHome(gameState, self, self.index)
        if len(self.aStarSearch(problem, self.GeneralHeuristic)) == 0:
          return 'Stop'
        else:
          return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
    else:
      if len(knowninvaders) == 0 and  self.lastEatenFoodPosition!=None and gameState.getAgentState(self.index).scaredTimer == 0:
        problem = SearchLastEatenFood(gameState,self,self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]
      if len(knowninvaders) > 0 and gameState.getAgentState(self.index).scaredTimer == 0:
        problem =  SearchInvaders(gameState,self,self.index)
        return self.aStarSearch(problem, gameState, self.GeneralHeuristic)[0]

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]
    return random.choice(bestActions)

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    features['dead'] = 0

    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0 and gameState.getAgentState(self.index).scaredTimer >0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = -1/min(dists)


    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    features['DistToBoundary'] = - self.distToHome(successor)
    return features

    if gameState.isOnRedTeam(self.index):
      if gameState.getAgentState(self.index) == (1,1):
        features['dead'] = 1
    else:
      if gameState.getAgentState(self.index) == (self.height-1,self.width-1):
        features['dead'] = 1


  def getWeights(self, gameState, action):
    return {'invaderDistance':1000,'onDefense': 200, 'stop': -100, 'reverse': -2,'DistToBoundary': 1,'dead':-10000}

