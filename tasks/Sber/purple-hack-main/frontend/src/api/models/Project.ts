import { RcFile } from 'antd/es/upload';

export interface CalendarInterval {
    recurrentStartDate: string | null;
    recurrentEndDate: string | null;
    isWorking: boolean;
    name: string | null;
    startDate: string;
    endDate: string;
    priority: number;
    recurrentWeekday: string | null;
    workException: string | null;
    overriddenWorkDayTimeInterval: unknown | null;
}

export interface Calendar {
    id: string;
    name: string;
    expanded: boolean;
    lastSaveDate: string;
    isOverriddenDateGrid: boolean;
    intervals: CalendarInterval[];
}

export interface ProjectTask {
    parentId?: string;
    children?: ProjectTask[];
    name: string;
    startDate: string;
    endDate: string;
    effort: number;
    effortUnit: string;
    duration: number;
    durationUnit: string;
    percentDone: number;
    schedulingMode: string;
    manuallyScheduled: boolean;
    effortDriven: boolean;
    parentIndex: number;
    expanded: boolean;
    inactive: boolean;
    critical: boolean;
    rootTask: boolean;
    priority: number;
    assignmentsUnitsSum: number;
    guid: string;
    id: string;
}

export interface Dependency {
    fromEvent: string;
    from: string;
    toEvent: string;
    to: string;
    lag: number;
    lagUnit: string;
    type: number;
    active: boolean;
    guid: string;
    id: string;
}

export interface Resource {
    name: string;
    projectRole: string;
    reservationType: string;
    reservationPercent: number;
    reservationStatus: string;
    projectRoleId: string;
    id: string;
}

export interface Assignment {
    event: string;
    resource: string;
    units: number;
    startDate: string;
    endDate: string;
    currentEffort: number;
    guid: string;
    id: string;
}

export interface Rows<T> {
    rows: T[];
}

export interface Project {
    id: string;
    projectId: string;
    name: string;
    isReadOnly: boolean;
    calendar: string;
    planType: string;
    lastModifyDate: string;
    lastModifyUser: string | null;
    publishDate: string | null;
    startDate: string;
}

export interface AccountingObject {
    id: string;
    name: string;
    startDate: string;
    endDate: string;
    type: string;
}

export interface ProjectInfo {
    requestId: string;
    project: Project;
    success: boolean;
    tasks: Rows<ProjectTask>;
    dependencies: Rows<Dependency>;
    calendars: Rows<Calendar>;
    resources: Rows<Resource>;
    assignments: Rows<Assignment>;
    timeRanges: Rows<unknown>;
    keyResults: Rows<unknown>;
    accountingObject: AccountingObject;
}

export interface UploadProjectFileParams {
    file: RcFile;
    price: number;
    duration: number;
    resource: number;
}
