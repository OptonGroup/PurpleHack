import { ProjectInfo, UploadProjectFileParams } from './models/Project';
import { API_URL } from '../config';
import axios from 'axios';

class TasksApiService {
    public async calculate({
        file,
        duration,
        price,
        resource,
    }: UploadProjectFileParams): Promise<ProjectInfo> {
        const formData = new FormData();

        formData.append('json_file', file);

        const response = await axios.post<ProjectInfo>(`${API_URL}/calculate`, formData, {
            params: { duration, price, resource },
            timeout: 120000,
        });

        return response.data;
    }
}

export const TasksApiServiceInstanse = new TasksApiService();
