# reinforcement-learning

CliffWalking-v0
 <img width="450" alt="image" src="https://github.com/ha-seungwon/reinforcement-learning/assets/74447373/1be6d143-859f-40cf-99bf-41c993fc6400">

Action Space 
  1) 0: move up
  2) 1: move right
   3) 2: move down
4) 3: move left

State Space: 4*12 = 48

Description
왼쪽 아래에서 시작하여 오른쪽 아래로 도착하는 것이 목표이며 낭떠러지에 빠지면 다시 시작점으로 돌아옵니다.

Rewards
  - Each step: -1
- Falling into the cliff: -100
( 추가적으로 학습 중reward에 대한 변화를 확실히 하고자 -100 -> -10 , final goal : +10으로 수정하였습니다.

FrozenLake-v1
 <img width="221" alt="image" src="https://github.com/ha-seungwon/reinforcement-learning/assets/74447373/fc7e12c3-9a0b-4e8b-b0d9-9e061a61f8f4">

Action Space
  - 0: LEFT
  - 1: DOWN
  - 2: RIGHT
- 3: UP

State Space: 4 * 4 = 16

Description
의자 있는 위치에서 시작하여 선물까지 도달하는게 목적이며 얼음 구덩이에 빠지면 다시 시작점으로 돌아옵니다.

Rewards
  - Reaching goal: +1
- Reaching hole: 0
(구덩이에 빠지는 경우 negative reward 가 존재하지 않아서 -10 으로 수정하였습니다.)


Taxi-v3
 <img width="290" alt="image" src="https://github.com/ha-seungwon/reinforcement-learning/assets/74447373/a844e24c-a8a9-4e18-849b-3273ae94293a">

Action Space
  - 0: move south
  - 1: move north
  - 2: move east
  - 3: move west
  - 4: pickup passenger
- 5: drop off passenger

State Space : 25(taxi position) * 5(passenger location) * 4(destination locations) = 500

Description
그리드 월드에는 R(빨간색), G(녹색), Y(노란색), B(파란색)으로 표시된 네 개의 지정된 위치가 있습니다. 에피소드가 시작되면 택시는 랜덤한 칸에서 시작하며 승객은 랜덤한 위치에 있습니다. 택시는 승객의 위치로 이동하여 승객을 태우고, 승객의 목적지(네 개의 지정된 위치 중 다른 하나)로 운전한 다음 승객을 내려줍니다. 승객을 내려주면 에피소드가 종료됩니다.

 Rewards
  - Each step: -1
  - Delivering passenger: +20
  - Illegal pickup/drop-off: -10

