COMMON_REGEX = [
    [
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?)\s(\s\d|\d{2})",
        "<DateText>",
    ],  # example : Jan 26 또는 February  3
    [r"\d{2}(\d{2}[\.|\-|\s]){2}\d{2}", "<Date>"],  # example : 2020-02-11
    [r"(\d{2}\:){2}\d{2}(\.\d{1,3})?", "<Time>"],  # example : 14:10:00.230
    [r"(\d+\.){3}\d+(\:\d+)?", "<IPv4>"],  # example : 178.146.52.4:5432
]
COMMON_DELIMITER = r"[\[\]()\'\={}\"\,/]"

### uAngel udrad
UANGEL_REGEX = [
    [r"\d{4}\|\s?(\d{2}\:){2}\d{2}\.\d{1,3}", "<Time>"],  # example : 1104| 14:10:00.230
    [r"(\w+\s)?([\w]|\/)+\.c:\d+\|", "<Source>"],  # example : | xlib ux_peer.c:1903|
    [r"(\d+\.){3}\d+(\:\d+)?", "<IPv4>"],  # example : 178.146.52.4:5432
]
UANGEL_DELIMITER = r"[()\'\={}\"\,/]"

### Linux syslog : /var/log/message
LINUX_SYSLOG_REGEX = [
    [
        r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|(Nov|Dec)(?:ember)?)\s(\s\d|\d{2})",
        "<Date>",
    ],  # example : Jan 26 또는 February  3
    [r"\D(\d{2}\:){2}\d{2}(\.\d{3}Z)?", "<Time>"],
    [r"\D\d{10}\.\d{1,4}\D", "<Timestamp>"],
    [r"Session\s\d{5}", "<SessionID>"],
    [r"[a-z|A-Z]+\s\d{4,5}\s?\:", "<ServiceID>"],
    [r"0x[0-9|a-f]{4,8}", "<MemAddress>"],
    [r"\d{4}\s[0-1]\d\s[0-3]\d", "<Date>"],
    [r"first\s\d+\slast\s\d+\sduration\s\d+\.\d+m?s", "<CheckPoint>"],
    [r"\w+\.go:\d+", "<Component>"],
    [r"ulid\s?(\d|[A-Z])+", "<ULID>"],
    [r"duration\s?\d+(\.\d+)?ms", "<Duration>"],
    [r"((min)|(max))t\s?\d+", "<MinMaxTime>"],
]
LINUX_SYSLOG_DELIMITER = r"[\[\-\]()\'\={}\"\,/]"

### Oracle listener log
ORA_LISTENER_REGEX = [
    [
        r"\d{2}(\d{2}\-){2}\d{2}T(\d{2}\:){2}\d{2}\.\d{6}\+\d{2}\:\d{2}",
        "<DateTime>",
    ],  # example : 2020-01-01T14:10:00.123456+09:00
    [
        r"\d{2}\-(?:JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\-\d{4}\s(\d{2}\:){2}\d{2}",
        "<DateTime>",
    ],  # example : 24-MAY-2019 09:46:24
    [r"TNS\-\d{5}\:?", "<TNSListener>"],  # example : TNS-12504
    [r"USER\s\S+", "<USER>"],
    [r"HOST\s\S+", "<HOST>"],
    [r"PORT\s\S+", "<PORT>"],
    [r"INSTANCE_NAME\s\S+", "<INSTNAME>"],
    [r"SERVICE_NAME\s\S+", "<SERVICENAME>"],
    [r"(version|VERSION)\s\S+", "<VERSION>"],
]

### Oracle alert log
ORA_ALERT_REGEX = [
    [
        r"\d{2}(\d{2}\-){2}\d{2}T(\d{2}\:){2}\d{2}\.\d{6}\+\d{2}\:\d{2}",
        "<DateTime>",
    ],  # example : 2020-01-01T14:10:00.123456+09:00
    [r"\d{2}(\d{2}[\.|\-|\s]){2}\d{2}", "<Date>"],  # example : 2020-02-11
    [r"(\d{2}\:){2}\d{2}", "<Time>"],  # example: 15:04:11
    [
        r"(\/[a-z|A-Z|0-9|ㄱ-힣|\-|\_|\%]+)+(.[a-z|A-Z]+)?",
        "<FilePath>",
    ],  # example : /u01/app/oracle/cfgtoollogs/dbca/orcl/initorclTempOMF.ora
    [r"0x[0-9|a-f]{4,8}", "<MemAddress>"],  # example : 0x7a1c6598
]
ORA_ALERT_DELIMITER = r"[\[\]()\'\={}\"\,]"

### JeusServer.log
JEUS_SERVER_REGEX = [
    # [r'\d{2}(\d{2}[\.|\-]){2}\d{2}\s+(\d{2}\:){2}\d{2}','<Date Time>'], # example : 2020.01.01 14:10:00 또는 2020-01-01 14:10:00
    [r"\d{2}(\d{2}[\.|\-|\s]){2}\d{2}", "<Date>"],  # example : 2020-02-11
    [r"(\d{2}\:){2}\d{2}\.\d{1,3}", "<Time>"],  # example : 14:10:00.230
    [r"(\d+\.){3}\d+(\:\d+)?", "<IPv4>"],  # example : 178.146.52.4:5432
    [r"이체\s금액\s\:\s\n+", "<Transfer Sum>"],  # example : 이체 금액 : 400
    [
        r"대외\s거래\s.+로\s대외\s이체\s종료\.",
        "<Transfer Summary>",
    ],  # example : 대외 거래 실패로 대외 이체 종료.
    [
        r"(성공)|(실패)|(\"(S|F)\")|(\W(S|F)\W)",
        "<S or F>",
    ],  # example : 성공 | 실패 | "S" | "F" | S, | F,
    # [r'\{((\"?\w+\"?(\=|\:)\"?(\w|\s|\!|[ㄱ-힣])+\"?\,)\s*)*(\"?\w+\"?(\=|\:)\"?(\w|\s|\!|[ㄱ-힣])+\"?)\}','<ObjectInfo>'], # example : {account_no=202002111410009, txn_inf=2000거래 실패!, txn_amount=9}
    [
        r"(\s((\d{8}|[\w\-]{1,})\sString)|(\d{13}\sLong))+",
        "<Parameters>",
    ],  # example : ojdmjxshwb String 20200219 String Gyeonggi-do String N String 1582040053681 Long 1582040053681 Long
    [r"\D\d{15}\D", "<Account No.>"],  # example : (space)221036457881234(space)
]

### ai-server log
AISERVER_REGEX = [
    # [r'\d{2}(\d{2}[\.|\-]){2}\d{2}\s*(\d{2}\:){2}\d{2}','<Date Time>'], # example : 2020-01-01 14:10:00 또는 2020-01-0114:10:00
    [r"\d{14}", "<DateTime>"],  # example : 20200307040653
    [r"\d{6}\s(\d{2}\:){2}\d{2}", "<DateTime>"],  # example : 200305 14:10:00
    [r"(\d+\.){3}\d+(\:\d+)?", "<IPv4>"],  # example : 178.146.52.4:5432
    [
        r"\[([a-z|A-Z|$|0-9|.])+\s*\:\s*\d+\]",
        "<Handler>",
    ],  # example : [com.exem.ai.dashboard.socket.StompHandler : 22]
    [r"\Wjava\.lang\.[A-Z][a-z]+\W", "<JavaType>"],  # example : java.lang.String
    [
        r"GenericMessage\s\[.*\]",
        "<GenericMessage>",
    ],  # example : GenericMessage [payload=byte[0], headers=...]
    [r"\{exem.*\}", "<exemModuleInfo>"],  # example : {exem_imxa_abclst_txn_3=...}
    [
        r"\{\s?(\w+\s?\:\s?[\w\d\s]+\,?)+\}",
        "<APIresponse>",
    ],  # example: { success :true data :false}
    [
        r"\.mapper\.\S+",
        "<Mapper>",
    ],  # example : ~.mapper.InstancePerformanceMapper.getInstance
    [r"id\s?\:\s?\d+", "<ID>"],  # example : id: 20142051
    [r"groupId=[\w\-]+", "<GroupID>"],  # example : groupId=ai-server
    [r"clientId=[\w\-]+", "<ClientID>"],  # example : groupId=ai-server
]
AISERVER_DELIMITER = r"[()\'\"\,\|/]"
