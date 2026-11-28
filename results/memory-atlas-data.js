// Reviewed summaries from MobileMem Text User 1; aliases replace original names and IDs.
const userMemory = {
  facts: [
    { id: 'badminton', entity: '羽毛球', type: 'SPORT', relation: 'HAS_INTEREST_IN', label: '关注的运动', databaseEvidence: 'e1', evidence: ['e1', 'e5', 'e6'] },
    { id: 'origin', entity: '通勤出发区', type: 'LOCATION', relation: 'COMMUTES_FROM', label: '通勤起点', evidence: ['e2'] },
    { id: 'business', entity: '商务区', type: 'LOCATION', relation: 'HAS_COMMUTE_DESTINATION', label: '通勤目的地', evidence: ['e2', 'e3'] },
    { id: 'person-a', entity: '同事 A', type: 'PERSON', relation: 'HAS_COLLEAGUE', label: '项目协作和拼场', evidence: ['e4', 'e5'] },
    { id: 'person-b', entity: '同事 B', type: 'PERSON', relation: 'HAS_COLLEAGUE', label: '项目协作和拼场', evidence: ['e4', 'e5'] },
    { id: 'gym', entity: '健身房', type: 'LOCATION', relation: 'GOES_TO', label: '周末运动计划', evidence: ['e5'] },
    { id: 'person-c', entity: '同事 C', type: 'PERSON', relation: 'HAS_COLLEAGUE', label: '帮忙确认球馆变更规则', evidence: ['e6'] },
    { id: 'food-photo', entity: '美食摄影', type: 'HOBBY', relation: 'LIKES', label: '结合探店记录美食', databaseEvidence: 'e9', evidence: ['e7', 'e8', 'e9'] },
  ],
  events: [
    { id: 'e1', date: '2024-06-01', title: '浏览羽毛球资料', text: '手机浏览记录保存了一篇关于羽毛球的词条。', source: 'Screen Memo', kind: '浏览' },
    { id: 'e2', date: '2024-06-04', title: '计划对比两条通勤路线', text: '同一出发区前往商务区，想对比两条路线的拥堵和停车。', source: '对话', kind: '计划', advice: { label: '通勤建议', text: '两条路线都记录耗时、拥堵和停车等待时间，再用同一套指标比较。' } },
    { id: 'e3', date: '2024-06-04', title: '试跑并记录拥堵与停车', text: '试跑其中一条路线后，记录了途中减速和到达商务区后的停车排队。', source: '对话', kind: '实测' },
    { id: 'e4', date: '2024-06-19', title: '整理项目确认项', text: '准备在行动看板上跟进多位同事的确认，同事 A、B 在名单中。', source: '对话', kind: '计划' },
    { id: 'e5', date: '2024-07-06', title: '制定一周运动安排', text: '计划工作日和同事 A、B 拼场打羽毛球，周末去健身房，并提前考虑加班时怎么调整。', source: '对话', kind: '计划' },
    { id: 'e6', date: '2024-07-10', title: '确认下班晚时的拼场备选', text: '同事 C 转达了球馆的最晚开球时间和改签限制。User 1 已经选好晚下班时的备选。', source: '对话', kind: '安排', advice: { label: '拼场提醒', text: '出发前对照已经定好的切换时间。晚于该时间就按备选到场，避免临时改签。' } },
    { id: 'e7', date: '2024-11-27', title: '建立探店和摄影待办', text: '在 Todo 里计划周末探索一家本地餐厅并拍照记录。该条记录显示未完成，无截止日期。', source: 'Todo', kind: '待办', advice: { label: '历史待办提醒', text: '先确认探店待办是否已经完成。如果还在进行，可以选一家餐厅并列出想拍的照片。' } },
    { id: 'e8', date: '2024-11-29', title: '探店待办标记完成', text: '同一个 Todo 的后续记录明确标记为完成，到这里不再显示该任务的未完成提醒。', source: 'Todo', kind: '完成' },
    { id: 'e9', date: '2024-12-01', title: '把拍照记录加进周末路线', text: '便签里计划把羽毛球、顺路探店和美食摄影安排在同一次出行。', source: 'Note', kind: '计划', advice: { label: '周末安排建议', text: '先确定打球结束时间，再挑顺路的一家餐厅，把拍照记录安排进同一次出行。' } },
  ],
};
