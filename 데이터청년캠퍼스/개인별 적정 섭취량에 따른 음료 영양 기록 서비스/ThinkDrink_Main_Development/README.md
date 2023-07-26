# think drink project!

트로젝트는 음료섭취에 관한 기록에 편의성을 제공하고 이용자의 대표적은 건강 지표를 이용해서 사용자에게 질병관리에 도움을 줄 수 있는 웹어플리케이션을 제공하고자 만들었습니다


# authors
project_name: think drink
project_excution_priod: 01.06.2022~30.08.2022
developers - 박진은, 박수정, 최예은, 최예주, 강재은

## 개발환경

mac Os 12.4
frame work: spring(maven)
RDMS: mySQL
front-engine: thymeleaf

## project_context
spring, JPA, bootstrap, and the others here is context of this project
├── HELP.md
├── README.md
├── biohealth2.iml
├── mvnw
├── mvnw.cmd
├── pom.xml
├── src
│   ├── main
│   │   ├── java
│   │   │   └── com
│   │   │       └── datacampus
│   │   │           └── biohealth2
│   │   │               ├── Biohealth2Application.java
│   │   │               ├── config
│   │   │               │   ├── AuditConfig.java
│   │   │               │   ├── AuditorAwareImpl.java
│   │   │               │   ├── CustomAuthenticationEntryPoint.java
│   │   │               │   ├── SecurityConfig.java
│   │   │               │   └── WebMvcConfig.java
│   │   │               ├── constant
│   │   │               │   ├── Activity.java
│   │   │               │   ├── Gender.java
│   │   │               │   ├── IceOrHot.java
│   │   │               │   └── Role.java
│   │   │               ├── controller
│   │   │               │   ├── BeverageInformationController.java
│   │   │               │   ├── BeverageIntakeController.java
│   │   │               │   ├── HealthInformationController.java
│   │   │               │   ├── HomeController.java
│   │   │               │   ├── IntakeBeverageInformation.java
│   │   │               │   ├── MemberController.java
│   │   │               │   └── ShowHealthInformationController.java
│   │   │               ├── dto
│   │   │               │   ├── BeverageFormDto.java
│   │   │               │   ├── BeverageImgDto.java
│   │   │               │   ├── BeverageInformationDto.java
│   │   │               │   ├── BeverageIntakeDto.java
│   │   │               │   ├── CafeInformationDto.java
│   │   │               │   ├── HealthInformationDto.java
│   │   │               │   ├── HealthInformationFormDto.java
│   │   │               │   ├── MemberDto.java
│   │   │               │   └── MemberFormDto.java
│   │   │               ├── entity
│   │   │               │   ├── BaseEntity.java
│   │   │               │   ├── BaseTimeEntity.java
│   │   │               │   ├── BeverageImg.java
│   │   │               │   ├── BeverageInformation.java
│   │   │               │   ├── BeverageIntake.java
│   │   │               │   ├── CafeInformation.java
│   │   │               │   ├── HealthInformation.java
│   │   │               │   └── Member.java
│   │   │               ├── repository
│   │   │               │   ├── BeverageImgRepository.java
│   │   │               │   ├── BeverageInformationRepository.java
│   │   │               │   ├── BeverageIntakeRepository.java
│   │   │               │   ├── CafeRepository.java
│   │   │               │   ├── HealthInformationRepository.java
│   │   │               │   └── MemberRepository.java
│   │   │               └── service
│   │   │                   ├── BeverageImgService.java
│   │   │                   ├── BeverageIntakeService.java
│   │   │                   ├── BeverageService.java
│   │   │                   ├── FileService.java
│   │   │                   ├── HealthInformationService.java
│   │   │                   └── MemberService.java
│   │   └── resources
│   │       ├── all_beverage.sql
│   │       ├── application-test.properties
│   │       ├── application.properties
│   │       ├── static


## compile

위 프로젝트를 컴파일할시 사용하는 spring MAVEN빌드 파일의 종속성을 다르시길 바랍니다. 또한 인텔리제이 이외의 IDE를 사용하는 경우에서는 iml파일을 삭제 후 실행하시기 바랍니다.

## Data
RDMS를 설정하고자 할때  user 의 정보와 Id의 local 환경에 맞게 설정해야 JPA의 원활한이요이 가능합니다. 또한 untitle.sql 파일에 더미 데이터로 담겨있는 정보들이 들어있기 때문에 이를 활용하기 위해서는 RDMS를 실행시키고 드라이버를 연결하고 이를 RDMS 내부에서 동작하게 만들거나 data.sql파일을 만들고 project properties를 변경하여 컴파일을 실행할때  데이터의 삽입이 일어나게 만들어야합니다. 본 git환경에서는 다른 사용자의 컴파일 환경을 알 수 없어 sql파일을 분리하여 업로드 했습니다.





