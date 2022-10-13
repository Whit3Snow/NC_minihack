"git 잘못 사용해서 날아감..."

10/13
Actor 모델(값이 distribution : softmax -> Categorical)
: 상태가 주어졌을 때, 행동을 결정
π(s,a)  
기대 출력 - Advantage 사용
Advantage : 예상했던 가치보다 얼마나 더 좋은 값인지 판단하는 값 
A(s,a)=Q(s,a)−V(s)   -->   A(s,a)≃R+γV(s′)−V(s)


Critic 모델(값이 하나로 나옴) 
: 상태의 가치를 평가
value V(s)  


replay buffer를 사용하지 않음
매 step마다 얻어진 s,a,r,s'을 통해 학습

Actor만 사용하면 값이 불안정해지기 때문에(즉, pg만 사용 시 불안정함)
가치 함수를 같이 사용해서 안정성을 높일 수 있음. 


ppo는 on policy라서 target network가 없는데, 이는 target policy와 이를 평가하는 target 가치 함수이다. 
