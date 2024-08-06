import React, { useState, useRef, useEffect } from 'react';
import moment from 'moment';
import './App.css'; // 업데이트된 CSS를 임포트합니다.

const ChatApp = () => {
    const [chatWindows, setChatWindows] = useState([{ id: 1, chats: [], showInfoBoxes: true }]);
    const [activeWindow, setActiveWindow] = useState(1);
    const [chatContents, setChatContents] = useState('');
    const [isLoading, setIsLoading] = useState(false);
    const chatInput = useRef(null);

    const nowTime = moment().format('MM-DD HH:mm:ss');

    const fetchBotResponse = async (message) => {
        setIsLoading(true);
        try {
            const response = await fetch('http://localhost:5000/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ input: message })
            });

            if (!response.ok) {
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

    const handleAddChat = async (message, isBot = false) => {
        // 메시지가 추가될 때 상태 업데이트
        setChatWindows(prev => {
            const currentWindow = prev.find(window => window.id === activeWindow);
            const updatedChats = [...currentWindow.chats, { no: currentWindow.chats.length + 1, chat: message, date: nowTime, isBot }];
    
            if (!isBot) {
                // 사용자의 메시지를 추가하고 봇의 응답을 받음
                return prev.map(window =>
                    window.id === activeWindow
                        ? { ...window, chats: updatedChats, showInfoBoxes: false }
                        : window
                );
            }
    
            // 봇의 응답이 있는 경우
            return prev.map(window =>
                window.id === activeWindow
                    ? { ...window, chats: updatedChats }
                    : window
            );
        });
    
        if (!isBot) {
            // 봇의 응답을 처리
            const botResponses = await fetchBotResponse(message);
            const botMessages = botResponses.map((item, index) => ({
                chat: `${item.menu} : ${item.text} (Score: ${item.score.toFixed(4)})`,
                date: nowTime,
                isBot: true
            }));
    
            setChatWindows(prev =>
                prev.map(window =>
                    window.id === activeWindow
                        ? { ...window, chats: [...prev.find(window => window.id === activeWindow).chats, ...botMessages] }
                        : window
                )
            );
        }
    };
    
    
    const pressEnter = (e) => {
        if (e.key === 'Enter' && chatContents.trim() !== '') {
            e.preventDefault();  // Enter 키 기본 동작 방지
            handleAddChat(chatContents);
        }
    };

    const addNewChatWindow = () => {
        const newId = chatWindows.length + 1;
        setChatWindows([...chatWindows, { id: newId, chats: [], showInfoBoxes: true }]);
        setActiveWindow(newId);
    };

    const handleBoxClick = (message) => {
        handleAddChat(message, true);
    };

    useEffect(() => {
        // `currentChats` 정의
        const currentWindow = chatWindows.find(window => window.id === activeWindow);
        const currentChats = currentWindow?.chats || [];

        // 채팅창을 아래로 스크롤
        scrollToBottom();
    }, [chatWindows, activeWindow]); // `chatWindows`와 `activeWindow`가 변경될 때 호출

    const scrollToBottom = () => {
        const { scrollHeight, clientHeight } = chatInput.current;
        chatInput.current.scrollTop = scrollHeight - clientHeight;
    };

    // 현재 활성화된 채팅창
    const currentWindow = chatWindows.find(window => window.id === activeWindow);
    const currentChats = currentWindow?.chats || [];
    const showInfoBoxes = currentWindow?.showInfoBoxes;

    return (
        <div className="ChatWrapper">
            <div className="sidebar">
                {chatWindows.map(window => (
                    <div 
                        key={window.id} 
                        className={`chatWindowButton ${window.id === activeWindow ? 'active' : ''}`}
                        onClick={() => setActiveWindow(window.id)}
                    >
                        채팅창 {window.id}
                    </div>
                ))}
                <button onClick={addNewChatWindow} className="newChatButton">새 채팅창</button>
            </div>
            <div className="chatContainer">
                {showInfoBoxes && (
                    <div className="infoBoxes">
                        <div className="infoBox" onClick={() => handleBoxClick('밥 추천하는 챗봇입니다.')}>밥 추천</div>
                        <div className="infoBox" onClick={() => handleBoxClick('칼로리 계산하는 챗봇입니다.')}>칼로리 계산</div>
                        <div className="infoBox" onClick={() => handleBoxClick('체중 조절 기간을 알려주는 챗봇입니다.')}>체중 조절 기간</div>
                    </div>
                )}
                <div className="chatList" ref={chatInput}>
                    {currentChats.map((item) => (
                        <div className={`chatContents ${item.isBot ? 'bot' : 'user'}`} key={item.no}>
                            <span className="chat">{item.chat}</span>
                            <span className="date">{item.date}</span>
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
