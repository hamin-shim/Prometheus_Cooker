import React, { useState, useRef, useEffect } from 'react';
import moment from 'moment';
import './App.css';

const ChatApp = () => {
    const [chatWindows, setChatWindows] = useState([{ id: 1, chats: [] }]);
    const [activeWindow, setActiveWindow] = useState(1);
    const [chatContents, setChatContents] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const [showBackground, setShowBackground] = useState(true);
    const [chatbotName, setChatbotName] = useState(''); 
    const [showCategories, setShowCategories] = useState(false);
    const [selectedCategory, setSelectedCategory] = useState('');
    const [dataType, setDataType] = useState('recommendation');
    const chatInput = useRef(null);

    const nowTime = moment().format('MM-DD HH:mm:ss');

    const fetchBotResponse = async (message) => {
        setIsLoading(true);
        try {
            const response = await fetch('http://127.0.0.1:5500/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    text: message, 
                    category: selectedCategory, 
                    type: dataType
                })
            });
            if (response.status !== 200) {
                throw new Error('Network response was not ok');
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error:', error);
            return [];
        } finally {
            setIsLoading(false);
        }
    };

    const handleAddChat = async (message, isBot = false, botResponses = []) => {
        setShowBackground(false);
        setChatWindows(prev => {
            const currentWindow = prev.find(window => window.id === activeWindow);
            const updatedChats = [...currentWindow.chats, { no: currentWindow.chats.length + 1, chat: message, date: nowTime, isBot }];
         
            return prev.map(window =>
                window.id === activeWindow
                    ? { ...window, chats: updatedChats }
                    : window
            );
        });
    
        if (!isBot && botResponses.length === 0) {
            const responses = await fetchBotResponse(message);
            if (dataType === 'recommendation') {
                handleAddChat('이런 메뉴는 어떠세요?', true, responses);
                handleAddChat("메뉴를 선택하면 더 많은 정보를 제공할게요!", true);
            } else if (dataType === 'calorie') {
                handleAddChat('이런 메뉴는 어떠세요?', true, responses);
                handleAddChat("이 중에서 칼로리가 궁금한 메뉴가 있다면 클릭해주세요!", true);
            }
        } else if (botResponses.length > 0) {
            botResponses.forEach((item, index) => {
                setChatWindows(prev =>
                    prev.map(window =>
                        window.id === activeWindow
                            ? { 
                                ...window, 
                                chats: [
                                    ...prev.find(window => window.id === activeWindow).chats, 
                                    {
                                        no: window.chats.length + 1 + index, 
                                        chat: (
                                            <span onClick={() => handleMenuClick(item)}>
                                                {`${item.menu} : ${item.text} (Score: ${item.score.toFixed(4)})`}
                                            </span>
                                        ),
                                        date: nowTime,
                                        isBot: true
                                    }
                                ] 
                            }
                            : window
                    )
                );
            });
        }
    };

    const handleMenuClick = (item) => {
        if (dataType === 'calorie') {
            const responseMessage = item.message;
            handleAddChat(responseMessage, true);
        } else {
            handleAddChat(`${item.menu}를 선택하셨습니다. 더 궁금한 점이 있으면 말씀해주세요!`, true);
        }
    };

    const pressEnter = (e) => {
        if (e.key === 'Enter' && chatContents.trim() !== '') {
            e.preventDefault();
            handleAddChat(chatContents);
        }
    };

    const addNewChatWindow = () => {
        const newId = chatWindows.length + 1;
        setChatWindows([...chatWindows, { id: newId, chats: [] }]);
        setActiveWindow(newId);
        setShowBackground(true);
        setChatbotName(''); 
        setShowCategories(false);
    };

    const handleWindowSwitch = (windowId) => {
        setActiveWindow(windowId);
        const currentWindow = chatWindows.find(window => window.id === windowId);
        // 채팅이 없으면 배경을 보여주고, 채팅이 있으면 배경을 숨깁니다.
        if (currentWindow.chats.length === 0) {
            setShowBackground(true);
        } else {
            setShowBackground(false);
        }
    };

    const handleInfoBoxClick = (message, botName, type) => { 
        setShowBackground(false);
        setChatbotName(botName); 
        setDataType(type); 
        handleAddChat(message, true);
        if (botName === '메뉴 추천 봇' || botName === '칼로리 계산 봇') {
            setShowCategories(true); 
        }
    };

    const handleCategoryClick = (category) => {
        setSelectedCategory(category);
        setShowCategories(false); 
        const message = (
            <span>
                {`${category} 카테고리를 선택하셨네요`}
                <br />
                <br />
                {"오늘은 어떤 음식이 땡기시나요?"}
                <br />
                <br />
                <span style={{ color: 'gray' }}>
                    {"ex ) 오늘 날씨가 추워서 뜨거운 음식이 먹고 싶어"}
                </span>
            </span>
        );
        handleAddChat(message, true); 
    };

    useEffect(() => {
        const currentWindow = chatWindows.find(window => window.id === activeWindow);
        if (currentWindow.chats.length === 0) {
            setShowBackground(true);
        } else {
            setShowBackground(false);
        }
        scrollToBottom();
        
    }, [chatWindows, activeWindow]);

    const scrollToBottom = () => {
        const { scrollHeight, clientHeight } = chatInput.current;
        chatInput.current.scrollTop = scrollHeight - clientHeight;
    };

    const currentWindow = chatWindows.find(window => window.id === activeWindow);
    const currentChats = currentWindow?.chats || [];

    return (
        <div className="ChatWrapper">
            <div className="sidebar">  
                {chatWindows.map(window => (
                    <div 
                        key={window.id} 
                        className={`chatWindowButton ${window.id === activeWindow ? 'active' : ''}`}
                        onClick={() => handleWindowSwitch(window.id)}
                    >
                        chat {window.id}
                    </div>
                ))}
                <button onClick={addNewChatWindow} className="newChatButton"> + 새 채팅 시작하기 </button>
            </div>
            <div className="chatContainer">
                {showBackground && (
                    <div className="chatBackground">
                        <img src="/img/main_img.png" alt="Chatbot logo" className="chatbotLogo" />
                        <p className="chatbotText">식단 추천 챗봇<br /><strong>오늘 뭐 먹지</strong>입니다</p>
                    </div>
                )}
                {showBackground && (
                    <div className="infoBoxes">
                        <div 
                            className="infoBox" 
                            onClick={() => handleInfoBoxClick("안녕하세요! 메뉴 추천 봇입니다. \n 카테고리를 선택하면 메뉴 추천을 도와드릴게요!", "메뉴 추천 봇", "recommendation")}
                        >
                            🍽️ <br /> 메뉴 추천받기
                        </div>
                        <div 
                            className="infoBox" 
                            onClick={() => handleInfoBoxClick("안녕하세요! 칼로리 계산 봇입니다. 카테고리를 선택하면 메뉴 추천 및 칼로리를 계산해드려요!", "칼로리 계산 봇", "calorie")}
                        >
                            🥗 <br /> 칼로리 계산하기
                        </div>
                        <div 
                            className="infoBox" 
                            onClick={() => handleInfoBoxClick("안녕하세요! 체중 조절 봇입니다. 현재 스펙을 알려주시면 체중 조절 기간을 알려드릴게요!", "체중 조절 챗봇", "weight_control")}
                        >
                            🔍 <br /> 체중 조절 기간 알아보기
                        </div>
                        
                    </div>
                )}
                <div className="chatList" ref={chatInput}>
                    {currentChats.map((item, index) => (
                        <div key={item.no}>
                            {item.isBot && (
                                <div className="chatbotName">
                                    <strong>{chatbotName}</strong>
                                </div>
                            )}
                            <div className={`chatContents ${item.isBot ? 'bot' : 'user'}`}>
                                {item.isBot && (
                                    <img src="/img/chatbot.png" alt="Chatbot" className="chatbotAvatar" />
                                )}
                                <span className="chat">{item.chat}</span>
                                <span className="date">{item.date}</span>
                            </div>
                            {index === currentChats.length - 1 && showCategories && (
                                <div className="categoryButtons">
                                    {["국물의 맛", "일품의 맛", "설탕의 맛", "밀의 맛", "쌀의 맛", "야채의 맛", "독특한 맛"].map(category => (
                                        <button
                                            key={category}
                                            className="categoryButton"
                                            onClick={() => handleCategoryClick(category)}
                                        >
                                            {category}
                                        </button>
                                    ))}
                                </div>
                            )}
                        </div>
                    ))}
                    {isLoading && <div className="chatContents bot"><span className="chat">...</span></div>}
                </div>
                <div className="chatArea">
                    <input
                        type="text"
                        value={chatContents}
                        onChange={(e) => setChatContents(e.target.value)}
                        onKeyPress={pressEnter}
                    />
                    <button onClick={() => {
                        if (chatContents.trim() !== '') {
                            handleAddChat(chatContents);
                        }
                    }}>전송</button>
                </div>
            </div>
        </div>
    );
};

export default ChatApp;
